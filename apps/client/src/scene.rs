//! Terrain scene for the client: load, generate, mesh, and expose vertices.
//!
//! The scene owns an authoritative world through [`WorldAuthority`]. Sections in
//! a square radius are generated deterministically and meshed once per terrain
//! revision. A mesh is rebuilt only when its section or one of its neighbours
//! changed, which is the rendering half of the revision contract the mesher
//! already enforces.

use std::collections::BTreeMap;

use torse_content::{BlockDefinition, Shape, compile, m0_fixture_pack};
use torse_render::{MeshData, MeshJob, MeshSample, RenderBlock, RenderBox, build_mesh};
use torse_sim::WorldAuthority;
use torse_sim::action::{ActionOrigin, ActionRequest, ActionService, ActionTarget, ActorContext};
use torse_sim::replica::{ReplicaChanges, ReplicaEntity, ReplicaSnapshot};
use torse_sim::{OrderingKey, Phase, block_actions};
use torse_types::{
    BlockPos, Compatibility, ContentKey, LocalPos, SectionPos, TransactionId, WorldId,
};
use torse_world::{
    BlockStateId, BlockStateRegistry, ComponentFields, ComponentSchemas, ComponentSet,
    ComponentValue, Position, Section, SnapshotCells, SnapshotSection, StateEntry, TerrainQuery,
    WorldBounds,
};
use torse_world_services::{GenerationParams, HillTerrain};

use crate::pick::{PickOutcome, cast_ray};

/// One vertex uploaded to the GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vertex {
    /// World-space position.
    pub position: [f32; 3],
    /// Face normal.
    pub normal: [f32; 3],
    /// Flat colour for the material.
    pub color: [f32; 3],
}

impl Vertex {
    /// Bytes per interleaved vertex.
    pub const STRIDE: usize = 36;

    fn write(&self, out: &mut Vec<u8>) {
        for value in self
            .position
            .iter()
            .chain(self.normal.iter())
            .chain(self.color.iter())
        {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// Terrain scene bound to one authoritative world.
pub struct Scene {
    authority: WorldAuthority,
    service: ActionService,
    principal: ContentKey,
    edit_counter: u64,
    walker: Option<torse_sim::player::WalkerState>,
    generator: Option<HillTerrain>,
    side: u32,
    radius: i64,
    /// Cached mesh plus the dependency revisions it was built against;
    /// a neighbour edit is detected by validating this stored job.
    meshes: BTreeMap<SectionPos, (MeshJob, MeshData)>,
    sections: Vec<SectionPos>,
}

/// One content pack for every client mode, matching the server demo assembly
/// so the handshake compatibility and the block colours agree everywhere.
fn content_pack() -> torse_content::ContentPack {
    let mut pack = m0_fixture_pack();
    block_actions::register_content(&mut pack);
    torse_inventory::register_content(&mut pack);
    pack
}

/// Build the registry, schemas, and handshake compatibility from the shared pack.
fn compile_content() -> Result<(BlockStateRegistry, ComponentSchemas, Compatibility), String> {
    let content =
        compile(content_pack()).map_err(|errors| format!("content failed: {errors:?}"))?;
    let registry =
        BlockStateRegistry::build(&content).map_err(|error| format!("registry failed: {error}"))?;
    let schemas = ComponentSchemas::build(&content);
    let compatibility = Compatibility {
        authority_schema: 4,
        rules_digest: content.authority_digest,
        content_digest: content.presentation_digest,
    };
    Ok((registry, schemas, compatibility))
}

impl Scene {
    /// Build a scene with a seeded hill terrain.
    pub fn new(radius: i64) -> Result<Self, String> {
        let (registry, schemas, compatibility) = compile_content()?;
        let stone = registry
            .default_state(&key("base:stone"))
            .ok_or_else(|| "missing base:stone".to_owned())?;
        let side = 16;
        let bounds = WorldBounds::new(0, 64).map_err(|error| error.to_string())?;
        let service = ActionService::bind(
            block_actions::descriptors(),
            block_actions::handlers(),
            &schemas,
        )
        .map_err(|error| format!("actions failed: {error}"))?;
        let authority =
            WorldAuthority::new(WorldId(1), registry, schemas, side, bounds, compatibility)
                .map_err(|error| error.to_string())?;
        let params = GenerationParams::new(side, bounds, stone, BlockStateId::EMPTY)
            .map_err(|error| error.to_string())?;
        let generator = HillTerrain::new(params, 7, 7, 9);
        let mut scene = Self {
            authority,
            service,
            principal: key("client:player"),
            edit_counter: 0,
            walker: None,
            generator: Some(generator),
            side,
            radius,
            meshes: BTreeMap::new(),
            sections: Vec::new(),
        };
        scene.load_around(SectionPos::ZERO)?;
        Ok(scene)
    }

    /// Build a scene from one authorized replica baseline.
    ///
    /// The network client has no generator: sections and entities come from
    /// the server projection only, and `load_around` keeps the subscribed
    /// section list as-is.
    pub fn from_snapshot(snapshot: &ReplicaSnapshot) -> Result<Self, String> {
        let (registry, schemas, compatibility) = compile_content()?;
        let service = ActionService::bind(
            block_actions::descriptors(),
            block_actions::handlers(),
            &schemas,
        )
        .map_err(|error| format!("actions failed: {error}"))?;
        let authority = WorldAuthority::new(
            WorldId(1),
            registry,
            schemas,
            snapshot.side,
            snapshot.bounds,
            compatibility,
        )
        .map_err(|error| error.to_string())?;
        let mut scene = Self {
            authority,
            service,
            principal: key("client:player"),
            edit_counter: 0,
            walker: None,
            generator: None,
            side: snapshot.side,
            radius: 0,
            meshes: BTreeMap::new(),
            sections: Vec::new(),
        };
        scene.install_snapshot(snapshot)?;
        Ok(scene)
    }

    /// Whether this scene mirrors a server projection instead of generating.
    pub const fn is_network(&self) -> bool {
        self.generator.is_none()
    }

    /// Handshake compatibility of the shared content pack.
    pub fn compatibility() -> Result<Compatibility, String> {
        Ok(compile_content()?.2)
    }

    /// Translate one replica baseline into local sections and entities.
    fn install_snapshot(&mut self, snapshot: &ReplicaSnapshot) -> Result<(), String> {
        let symbols = self.resolve_states(&snapshot.states)?;
        for section in &snapshot.sections {
            self.install_replica_section(section, &symbols)?;
        }
        self.sections = snapshot.sections.iter().map(|s| s.pos).collect();
        for entity in &snapshot.entities {
            self.install_replica_entity(entity)?;
        }
        Ok(())
    }

    /// Map the snapshot's runtime state table into local registry IDs.
    fn resolve_states(
        &self,
        states: &[StateEntry],
    ) -> Result<BTreeMap<BlockStateId, BlockStateId>, String> {
        let mut symbols = BTreeMap::new();
        symbols.insert(BlockStateId::EMPTY, BlockStateId::EMPTY);
        for entry in states {
            let local = registry_state(self.authority.registry(), &entry.block, &entry.values)?;
            symbols.insert(entry.id, local);
        }
        Ok(symbols)
    }

    /// Install or replace one section from replica cells.
    fn install_replica_section(
        &mut self,
        section: &SnapshotSection,
        symbols: &BTreeMap<BlockStateId, BlockStateId>,
    ) -> Result<(), String> {
        let side = self.side;
        let mut built = Section::new(side, BlockStateId::EMPTY)
            .map_err(|error| format!("section failed: {error}"))?;
        let write = |built: &mut Section, index: usize, state: BlockStateId| {
            let side = side as usize;
            let x = (index % side) as u32;
            let z = ((index / side) % side) as u32;
            let y = (index / (side * side)) as u32;
            built
                .set(LocalPos { x, y, z }, state)
                .map_err(|error| format!("cell failed: {error}"))
        };
        match &section.cells {
            SnapshotCells::Uniform(state) => {
                let local = *symbols.get(state).ok_or("unknown replica state")?;
                if !local.is_empty() {
                    // `Section::new` fills every cell; only real blocks need
                    // per-cell writes.
                    for index in 0..(side as usize).pow(3) {
                        write(&mut built, index, local)?;
                    }
                }
            }
            SnapshotCells::Dense(cells) => {
                if cells.len() != (side as usize).pow(3) {
                    return Err("dense section has the wrong length".to_owned());
                }
                for (index, state) in cells.iter().enumerate() {
                    let local = *symbols.get(state).ok_or("unknown replica state")?;
                    if !local.is_empty() {
                        write(&mut built, index, local)?;
                    }
                }
            }
        }
        let store = self.authority.store_mut();
        let _ = store.unload(section.pos);
        store
            .request(section.pos)
            .map_err(|error| error.to_string())?;
        store
            .install(section.pos, built)
            .map_err(|error| error.to_string())?;
        store
            .finish_load(section.pos)
            .map_err(|error| error.to_string())?;
        // A reinstalled section can reuse the previous revision number, so the
        // cached mesh must go or the change would render as a no-op.
        self.meshes.remove(&section.pos);
        Ok(())
    }

    /// Insert or replace one entity from a replica record, keeping its
    /// persistent identity.
    fn install_replica_entity(&mut self, entity: &ReplicaEntity) -> Result<(), String> {
        if let Some(handle) = self.authority.entities().handle_of(entity.id) {
            let _ = self.authority.entities_mut().despawn(handle);
        }
        let position = Position::from_bits(entity.position_bits);
        let mut components = ComponentSet::new();
        let schemas = self.authority.schemas().clone();
        for (schema, fields) in &entity.components {
            components
                .insert(&schemas, schema, fields.clone())
                .map_err(|error| format!("component failed: {error}"))?;
        }
        self.authority
            .entities_mut()
            .insert_restored(entity.id, position, components)
            .map_err(|error| format!("entity failed: {error}"))?;
        Ok(())
    }

    /// Apply one replica increment: replaced sections and entities, removals.
    pub fn apply_changes(&mut self, changes: &ReplicaChanges) -> Result<(), String> {
        let symbols = self.resolve_states(&changes.states)?;
        for pos in &changes.removed_sections {
            let _ = self.authority.store_mut().unload(*pos);
            self.meshes.remove(pos);
            self.sections.retain(|item| item != pos);
        }
        for section in &changes.sections {
            self.install_replica_section(section, &symbols)?;
            if !self.sections.contains(&section.pos) {
                self.sections.push(section.pos);
            }
        }
        for id in &changes.removed_entities {
            if let Some(handle) = self.authority.entities().handle_of(*id) {
                let _ = self.authority.entities_mut().despawn(handle);
            }
        }
        for entity in &changes.entities {
            self.install_replica_entity(entity)?;
        }
        Ok(())
    }

    /// Authoritative world.
    pub const fn authority(&self) -> &WorldAuthority {
        &self.authority
    }

    /// Classify one cell for picking: solid, empty, or outside the loaded world.
    fn solid_at(&self, pos: BlockPos) -> Option<bool> {
        match self.authority.store().query(pos) {
            TerrainQuery::Ready(state) => Some(!state.is_empty()),
            TerrainQuery::NotLoaded | TerrainQuery::OutOfBounds => None,
        }
    }

    /// Cast a ray through loaded terrain and report the first solid cell.
    pub fn pick(&self, origin: [f32; 3], direction: [f32; 3], max_distance: f32) -> PickOutcome {
        cast_ray(|pos| self.solid_at(pos), origin, direction, max_distance)
    }

    /// Spawn the walk-mode player entity at a world position.
    ///
    /// The windowed client is a trusted local host, so it installs the player
    /// directly through the entity store the same way the server bootstrap
    /// installs its fixtures; movement still goes through transactions.
    pub fn spawn_walker(&mut self, at: Position) -> Result<(), String> {
        let handle = self
            .authority
            .entities_mut()
            .spawn(at)
            .map_err(|error| format!("spawn failed: {error}"))?;
        self.walker = Some(torse_sim::player::WalkerState::new(handle));
        Ok(())
    }

    /// Walker position, if walk mode has a live entity.
    pub fn walker_position(&self) -> Option<Position> {
        let walker = self.walker?;
        self.authority
            .entities()
            .get(walker.handle)
            .map(|entity| entity.position)
    }

    /// Integrate one walk tick through the shared player domain; see the
    /// movement contract. Rejections drop the tick and keep the entity at
    /// its committed position.
    pub fn step_walker(
        &mut self,
        wish: [f64; 3],
        jump: bool,
        dt: f32,
    ) -> Result<(Position, bool), String> {
        let Some(mut walker) = self.walker else {
            return Err("walk mode has no entity".to_owned());
        };
        self.edit_counter += 1;
        let counter = self.edit_counter;
        let outcome = torse_sim::player::walker_step(
            &mut self.authority,
            &mut walker,
            wish,
            jump,
            dt,
            &self.principal,
            TransactionId(u128::from(counter)),
        );
        self.walker = Some(walker);
        match outcome {
            Ok(result) => Ok(result),
            Err(torse_sim::player::StepError::EntityGone) => {
                Err("walker entity is gone".to_owned())
            }
            Err(torse_sim::player::StepError::Fault(fault)) => {
                Err(format!("commit fault: {fault:?}"))
            }
        }
    }

    /// Plan and commit one place or break action through the action service.
    ///
    /// The windowed client is a trusted local host: it grants the player the
    /// edit permission and the single target cell for this request, then lets
    /// the service authorize, validate, and commit like any other adapter.
    /// The error string names the rejection, it is diagnostics only.
    pub fn apply_edit(
        &mut self,
        place: bool,
        block: &ContentKey,
        target: BlockPos,
    ) -> Result<(), String> {
        self.edit_counter += 1;
        let counter = self.edit_counter;
        let action = if place {
            key("world:action/place")
        } else {
            key("world:action/break")
        };
        let mut parameters = ComponentFields::new();
        if place {
            parameters.insert(
                key("world:field/block"),
                ComponentValue::Content(block.clone()),
            );
        }
        let request = ActionRequest {
            action,
            api_version: 1,
            origin: ActionOrigin::Ui,
            target: ActionTarget::Block(target),
            parameters,
            transaction: TransactionId(u128::from(counter)),
            ordering: OrderingKey {
                phase: Phase::Interaction,
                rule_priority: 0,
                stable_task_key: counter,
                local_sequence: 0,
            },
        };
        let actor = ActorContext::new(self.principal.clone())
            .permission(key("world:permission/edit"))
            .target(ActionTarget::Block(target));
        let output = self
            .service
            .plan(&self.authority, &actor, Phase::Interaction, &request)
            .map_err(|code| format!("{code:?}"))?;
        let torse_sim::action::ActionOutput::Mutation(transaction) = output else {
            return Err("place and break must be mutations".to_owned());
        };
        self.authority
            .commit(&transaction)
            .map_err(|fault| format!("{fault:?}"))?;
        Ok(())
    }

    /// Loaded section coordinates.
    pub fn sections(&self) -> &[SectionPos] {
        &self.sections
    }

    /// Number of cached meshes.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Load sections inside the radius around a centre and unload the rest.
    ///
    /// Network scenes keep the server's subscribed sections; streaming would
    /// fabricate terrain the server does not have.
    pub fn load_around(&mut self, center: SectionPos) -> Result<(), String> {
        if self.generator.is_none() {
            return Ok(());
        }
        let mut wanted = Vec::new();
        for x in -self.radius..=self.radius {
            for z in -self.radius..=self.radius {
                let sx = center
                    .x
                    .checked_add(x)
                    .ok_or("streaming coordinate overflow")?;
                let sz = center
                    .z
                    .checked_add(z)
                    .ok_or("streaming coordinate overflow")?;
                let pos = SectionPos::new(sx, 0, sz);
                pos.to_block(LocalPos { x: 0, y: 0, z: 0 }, self.side)
                    .map_err(|error| error.to_string())?;
                wanted.push(pos);
            }
        }

        for pos in &wanted {
            if self.authority.store().phase(*pos).is_none() {
                let store = self.authority.store_mut();
                store.request(*pos).map_err(|error| error.to_string())?;
                let section = self
                    .generator
                    .as_mut()
                    .expect("generator checked above")
                    .generate(*pos)
                    .map_err(|error| error.to_string())?;
                store
                    .install(*pos, section)
                    .map_err(|error| error.to_string())?;
                store.finish_load(*pos).map_err(|error| error.to_string())?;
            }
        }

        let wanted_set = wanted.clone();
        for pos in self.sections.clone() {
            if !wanted_set.contains(&pos) {
                let store = self.authority.store_mut();
                let _ = store.unload(pos);
                self.meshes.remove(&pos);
            }
        }
        self.sections = wanted;
        Ok(())
    }

    /// Rebuild meshes whose section or neighbour revision changed.
    ///
    /// The cache holds the dependency job captured at build time; validating
    /// that stored job against the live store detects neighbour edits too.
    /// Capturing a fresh job here instead would compare the world with itself
    /// and never notice a neighbour change.
    ///
    /// Returns how many meshes were rebuilt, so callers can skip a GPU upload
    /// when nothing changed.
    pub fn refresh_meshes(&mut self) -> usize {
        let store = self.authority.store();
        let registry = self.authority.registry();
        let mut rebuilt = 0usize;
        for pos in &self.sections {
            let unchanged = self
                .meshes
                .get(pos)
                .is_some_and(|(job, _)| job.validate(|query| store.revision(query)));
            if unchanged {
                continue;
            }
            let job = MeshJob::capture(self.side, *pos, |query| store.revision(query));
            let Some(_revision) = job.section_revision else {
                continue;
            };
            let mesh = build_mesh(self.side, *pos, |query| sample(store, registry, query));
            self.meshes.insert(*pos, (job, mesh));
            rebuilt += 1;
        }
        rebuilt
    }

    /// Face count of one cached mesh, for tests.
    #[cfg(test)]
    pub fn mesh_face_count(&self, pos: SectionPos) -> Option<usize> {
        self.meshes.get(&pos).map(|(_, mesh)| mesh.face_count())
    }

    /// Section containing a world-space position on the section layer.
    pub fn center_section(&self, position: [f32; 3]) -> SectionPos {
        SectionPos::new(
            (position[0] as i64).div_euclid(self.side as i64),
            0,
            (position[2] as i64).div_euclid(self.side as i64),
        )
    }

    /// Flatten every cached mesh into one interleaved vertex and index buffer,
    /// then append one coloured box per stored entity so mirrored bodies are
    /// visible even before real entity art exists.
    pub fn geometry(&self) -> (Vec<Vertex>, Vec<u32>, usize) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut faces = 0usize;
        for (pos, (_, mesh)) in &self.meshes {
            let offset = [
                pos.x as f32 * self.side as f32,
                0.0,
                pos.z as f32 * self.side as f32,
            ];
            let base = vertices.len() as u32;
            for (index, position) in mesh.positions.iter().enumerate() {
                let material = mesh.materials[index / 4];
                vertices.push(Vertex {
                    position: [
                        position[0] + offset[0],
                        position[1] + offset[1],
                        position[2] + offset[2],
                    ],
                    normal: mesh.normals[index],
                    color: material_color(self.authority.registry(), material),
                });
            }
            indices.extend(mesh.indices.iter().map(|value| base + value));
            faces += mesh.face_count();
        }
        for (handle, entity) in self.authority.entities().iter() {
            // The walker is the camera; drawing its body would fill the view.
            if Some(handle) == self.walker.map(|walker| walker.handle) {
                continue;
            }
            faces += push_entity_box(
                &mut vertices,
                &mut indices,
                [
                    entity.position.x as f32,
                    entity.position.y as f32,
                    entity.position.z as f32,
                ],
            );
        }
        (vertices, indices, faces)
    }

    /// Encode vertices into the interleaved GPU layout.
    pub fn vertex_bytes(vertices: &[Vertex]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(vertices.len() * Vertex::STRIDE);
        for vertex in vertices {
            vertex.write(&mut bytes);
        }
        bytes
    }

    /// Camera position standing on the terrain at the scene centre.
    pub fn eye(&self) -> [f32; 3] {
        let x = 8;
        let z = 8;
        for y in (0..64).rev() {
            if let TerrainQuery::Ready(state) = self.authority.store().query(BlockPos::new(x, y, z))
                && !state.is_empty()
            {
                return [x as f32 + 0.5, y as f32 + 2.4, z as f32 + 0.5];
            }
        }
        [8.5, 12.0, 8.5]
    }
}

/// Resolve stable content into a local runtime state ID.
fn registry_state(
    registry: &BlockStateRegistry,
    block: &ContentKey,
    values: &[u16],
) -> Result<BlockStateId, String> {
    registry
        .state_of(block, values)
        .ok_or_else(|| format!("unresolved replica content {block}"))
}

/// Append one boxy body for an entity; returns the face count added.
fn push_entity_box(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, center: [f32; 3]) -> usize {
    const HALF: f32 = 0.35;
    let color = [0.78, 0.55, 0.35];
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [HALF, -HALF, -HALF], [HALF, HALF, HALF]),
        ([-1.0, 0.0, 0.0], [-HALF, HALF, -HALF], [-HALF, -HALF, HALF]),
        ([0.0, 1.0, 0.0], [-HALF, HALF, -HALF], [HALF, HALF, HALF]),
        ([0.0, -1.0, 0.0], [-HALF, -HALF, HALF], [HALF, -HALF, -HALF]),
        ([0.0, 0.0, 1.0], [-HALF, -HALF, HALF], [HALF, HALF, HALF]),
        ([0.0, 0.0, -1.0], [-HALF, HALF, -HALF], [HALF, -HALF, -HALF]),
    ];
    let mut added = 0usize;
    for (normal, min_corner, max_corner) in faces {
        let (nx, ny, nz) = (normal[0], normal[1], normal[2]);
        let (lo, hi) = (min_corner, max_corner);
        let corners = [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
        ];
        let base = vertices.len() as u32;
        for corner in corners {
            vertices.push(Vertex {
                position: [
                    center[0] + corner[0],
                    center[1] + corner[1],
                    center[2] + corner[2],
                ],
                normal: [nx, ny, nz],
                color,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        added += 2;
    }
    added
}

fn key(value: &str) -> ContentKey {
    ContentKey::parse(value).expect("fixture key is valid")
}

fn sample(
    store: &torse_world::SectionStore,
    registry: &BlockStateRegistry,
    pos: BlockPos,
) -> MeshSample {
    match store.query(pos) {
        TerrainQuery::Ready(state) => {
            if state.is_empty() {
                return MeshSample::Ready(empty_block());
            }
            let Some(definition) = registry.definition(state) else {
                return MeshSample::Ready(empty_block());
            };
            MeshSample::Ready(RenderBlock {
                bounds: render_box(definition),
                occlusion: definition.occlusion_profile.face_mask,
                material: state.0,
            })
        }
        TerrainQuery::NotLoaded => MeshSample::NotLoaded,
        TerrainQuery::OutOfBounds => MeshSample::OutOfBounds,
    }
}

const fn empty_block() -> RenderBlock {
    RenderBlock {
        bounds: None,
        occlusion: 0,
        material: 0,
    }
}

fn render_box(definition: &BlockDefinition) -> Option<RenderBox> {
    match &definition.render_profile.shape {
        Shape::FullCube => Some(RenderBox::FULL_CUBE),
        Shape::LowerSlab => Some(RenderBox {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 0.5, 1.0],
        }),
        Shape::Panel => Some(RenderBox {
            min: [0.45, 0.0, 0.0],
            max: [0.55, 1.0, 1.0],
        }),
        Shape::Corner | Shape::Stair => Some(RenderBox::FULL_CUBE),
        Shape::Box(bounds) => Some(RenderBox {
            min: bounds.min,
            max: bounds.max,
        }),
    }
}

fn material_color(registry: &BlockStateRegistry, material: u16) -> [f32; 3] {
    let Some(path) = registry
        .definition_key(BlockStateId(material))
        .map(|key| key.path())
    else {
        return [0.70, 0.70, 0.70];
    };
    match path {
        "stone" => [0.62, 0.61, 0.58],
        "glass_slab" => [0.62, 0.78, 0.82],
        "plant" => [0.36, 0.55, 0.32],
        _ => [0.70, 0.70, 0.70],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torse_types::EntityId;

    #[test]
    fn scene_generates_and_meshes_its_radius() {
        let mut scene = Scene::new(1).expect("scene builds");
        assert_eq!(scene.sections().len(), 9);
        assert_eq!(scene.refresh_meshes(), 9);
        assert_eq!(scene.mesh_count(), 9);
        // Nothing changed, so the second pass rebuilds nothing.
        assert_eq!(scene.refresh_meshes(), 0);
        let (vertices, indices, faces) = scene.geometry();
        assert!(!vertices.is_empty());
        assert!(!indices.is_empty());
        assert!(faces > 0);
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn neighbour_edit_invalidates_the_border_mesh() {
        let mut scene = Scene::new(1).expect("scene builds");
        scene.refresh_meshes();
        let stone = key("base:stone");
        // X sits in section (0,0,0) against the border, exposing a +x face
        // into neighbour (1,0,0).
        scene
            .apply_edit(true, &stone, BlockPos::new(15, 15, 1))
            .expect("place X");
        scene.refresh_meshes();
        let faces_before = scene.mesh_face_count(SectionPos::ZERO).expect("cached");

        // Y lands in the neighbour and occludes that face; the border
        // section's own revision does not change.
        scene
            .apply_edit(true, &stone, BlockPos::new(16, 15, 1))
            .expect("place Y");
        let rebuilt = scene.refresh_meshes();
        assert!(rebuilt >= 2, "neighbour and border section must rebuild");

        let faces_after = scene.mesh_face_count(SectionPos::ZERO).expect("cached");
        assert_eq!(
            faces_before - faces_after,
            1,
            "the occluded border face is gone"
        );
    }

    #[test]
    fn streaming_recenters_without_leaking_sections() {
        let mut scene = Scene::new(1).expect("scene builds");
        scene.refresh_meshes();
        let before = scene.geometry();
        assert!(!before.0.is_empty());

        scene.load_around(SectionPos::new(3, 0, 0)).expect("load");
        assert_eq!(scene.sections().len(), 9);
        // The unloaded column's meshes are gone; the new ones are built.
        assert!(scene.refresh_meshes() > 0);
        assert_eq!(scene.mesh_count(), 9);
        let after = scene.geometry();
        assert!(!after.0.is_empty());

        // Returning to the origin reloads the same deterministic terrain.
        scene.load_around(SectionPos::ZERO).expect("load back");
        scene.refresh_meshes();
        assert_eq!(scene.sections().len(), 9);
        assert_eq!(scene.mesh_count(), 9);
        assert_eq!(scene.geometry().0.len(), before.0.len());
    }

    #[test]
    fn scrolling_once_then_flying_keeps_streaming_bounded() {
        let mut scene = Scene::new(0).unwrap();
        let mut camera = crate::camera::FlyCamera::new(scene.eye());
        let mut input = crate::input::InputState::new();
        input.key(
            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyW),
            winit::event::ElementState::Pressed,
        );
        input.wheel(winit::event::MouseScrollDelta::LineDelta(0.0, 1.0));
        let mut center = scene.center_section(camera.eye);
        let mut streamed = 0;
        for _ in 0..3600 {
            let (motion, speed) = input.take_input(camera.speed);
            camera.speed = speed;
            camera.advance(motion, 1.0 / 60.0);
            assert_eq!(camera.speed, 15.0);
            assert!(camera.eye.iter().all(|v| v.is_finite()));
            let next = scene.center_section(camera.eye);
            if next != center {
                scene.load_around(next).unwrap();
                center = next;
                streamed += 1;
                assert_eq!(scene.sections().len(), 1);
            }
        }
        assert!(
            streamed > 40,
            "exercise repeated section loading after the wheel event"
        );
    }

    #[test]
    fn streaming_rejects_overflow_before_changing_the_scene() {
        let mut scene = Scene::new(1).unwrap();
        let before = scene.sections().to_vec();
        for center in [
            SectionPos::new(i64::MAX, 0, 0),
            SectionPos::new(0, 0, i64::MIN),
        ] {
            assert!(scene.load_around(center).is_err());
            assert_eq!(scene.sections(), before);
        }
    }

    #[test]
    fn center_section_maps_positions_including_negatives() {
        let scene = Scene::new(0).expect("scene builds");
        assert_eq!(scene.center_section([0.0, 5.0, 0.0]), SectionPos::ZERO);
        assert_eq!(
            scene.center_section([17.5, 5.0, -1.0]),
            SectionPos::new(1, 0, -1)
        );
        assert_eq!(
            scene.center_section([-16.0, 5.0, -17.0]),
            SectionPos::new(-1, 0, -2)
        );
    }

    #[test]
    fn vertex_bytes_match_the_documented_stride() {
        let vertices = [Vertex {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 1.0, 0.0],
            color: [0.5, 0.5, 0.5],
        }];
        let bytes = Scene::vertex_bytes(&vertices);
        assert_eq!(bytes.len(), Vertex::STRIDE);
    }

    fn query(scene: &Scene, pos: BlockPos) -> bool {
        matches!(
            scene.authority().store().query(pos),
            TerrainQuery::Ready(state) if !state.is_empty()
        )
    }

    /// Find a column whose surface leaves a loaded, empty cell above it.
    /// Only the `y = 0` section layer is loaded, so test cells must stay below
    /// 16.
    fn air_above_surface(scene: &Scene) -> BlockPos {
        for x in 0..12i64 {
            for z in 0..12i64 {
                for y in (0..14i64).rev() {
                    let pos = BlockPos::new(x, y, z);
                    if query(scene, pos) {
                        return BlockPos::new(x, y + 1, z);
                    }
                }
            }
        }
        panic!("no air cell found above the surface");
    }

    #[test]
    fn apply_edit_places_and_breaks_through_the_action_service() {
        let mut scene = Scene::new(1).expect("scene builds");
        let target = air_above_surface(&scene);
        let surface = BlockPos::new(target.x, target.y - 1, target.z);
        assert!(!query(&scene, target));
        assert!(query(&scene, surface));

        let stone = key("base:stone");
        scene
            .apply_edit(true, &stone, target)
            .expect("place commits");
        assert!(query(&scene, target));

        // Placing into the now occupied cell is rejected without a commit.
        let again = scene.apply_edit(true, &stone, target);
        assert!(again.is_err());

        scene
            .apply_edit(false, &stone, target)
            .expect("break commits");
        assert!(!query(&scene, target));

        // Breaking air is a conflict, not a silent no-op.
        let empty = scene.apply_edit(false, &stone, target);
        assert!(empty.is_err());
    }

    #[test]
    fn apply_edit_rejects_out_of_bounds_targets() {
        let mut scene = Scene::new(1).expect("scene builds");
        let stone = key("base:stone");
        assert!(
            scene
                .apply_edit(true, &stone, BlockPos::new(0, 64, 0))
                .is_err()
        );
        assert!(
            scene
                .apply_edit(true, &stone, BlockPos::new(0, -1, 0))
                .is_err()
        );
        // Above the loaded layer the target is not loaded, which is also a
        // rejection rather than a silent drop.
        assert!(
            scene
                .apply_edit(true, &stone, BlockPos::new(0, 40, 0))
                .is_err()
        );
    }

    #[test]
    fn pick_finds_the_surface_below_the_camera() {
        let scene = Scene::new(1).expect("scene builds");
        // Start inside the loaded layer so the ray can walk down to the hill.
        let outcome = scene.pick([8.5, 15.5, 8.5], [0.0, -1.0, 0.0], 128.0);
        let PickOutcome::Hit(hit) = outcome else {
            panic!("expected the terrain surface");
        };
        assert_eq!((hit.block.x, hit.block.z), (8, 8));
        assert!(hit.block.y < 16);
        assert!(hit.distance < 16.0);
    }

    #[test]
    fn pick_stops_at_unloaded_sections() {
        let scene = Scene::new(1).expect("scene builds");
        // Radius 1 leaves everything past 31 blocks unloaded on the x axis.
        let outcome = scene.pick([30.5, 40.0, 8.5], [1.0, 0.0, 0.0], 256.0);
        assert_eq!(outcome, PickOutcome::Miss);
    }

    fn stone_layer_section(pos: SectionPos) -> SnapshotSection {
        let side = 16u32;
        let mut cells = vec![BlockStateId::EMPTY; side.pow(3) as usize];
        for x in 0..side {
            for z in 0..side {
                let index = x as usize + side as usize * z as usize;
                cells[index] = BlockStateId(7);
            }
        }
        SnapshotSection {
            pos,
            cells: SnapshotCells::Dense(cells),
        }
    }

    fn replica_snapshot() -> ReplicaSnapshot {
        ReplicaSnapshot {
            side: 16,
            bounds: WorldBounds::new(0, 32).expect("bounds"),
            states: vec![StateEntry {
                id: BlockStateId(7),
                block: key("base:stone"),
                values: Vec::new(),
            }],
            sections: vec![
                stone_layer_section(SectionPos::ZERO),
                stone_layer_section(SectionPos::new(1, 0, 0)),
            ],
            entities: vec![ReplicaEntity {
                id: EntityId(9),
                position_bits: Position::new(2.5, 2.0, 2.0).to_bits(),
                components: BTreeMap::new(),
            }],
        }
    }

    #[test]
    fn network_scene_installs_a_replica_baseline() {
        let mut scene = Scene::from_snapshot(&replica_snapshot()).expect("scene");
        assert!(scene.is_network());
        // The symbol table resolved remote runtime ID 7 into local stone.
        assert!(query(&scene, BlockPos::new(3, 0, 4)));
        assert!(!query(&scene, BlockPos::new(3, 1, 4)));
        // The mirrored entity landed in the client entity store.
        assert_eq!(scene.authority().entities().len(), 1);
        let stored = scene.authority().entities().get(
            scene
                .authority()
                .entities()
                .handle_of(EntityId(9))
                .expect("entity"),
        );
        assert_eq!(stored.expect("entity").position.y, 2.0);
        // Meshes build from the mirrored sections alone.
        assert_eq!(scene.refresh_meshes(), 2);
        let (vertices, indices, _) = scene.geometry();
        assert!(!vertices.is_empty() && !indices.is_empty());
        // Streaming must not fabricate terrain around the mirror.
        scene.load_around(SectionPos::new(5, 0, 5)).expect("noop");
        assert_eq!(scene.sections().len(), 2);
    }

    #[test]
    fn network_scene_applies_a_replica_increment() {
        let mut scene = Scene::from_snapshot(&replica_snapshot()).expect("scene");
        scene.refresh_meshes();
        // Carve one hole by replacing the section without its stone at x=3,z=4.
        let mut cells = vec![BlockStateId(7); 16usize.pow(3)];
        for y in 0..16usize {
            for x in 0..16usize {
                for z in 0..16usize {
                    let index = x + 16 * (z + 16 * y);
                    cells[index] = if y == 0 && x != 3 && z != 4 {
                        BlockStateId(7)
                    } else {
                        BlockStateId::EMPTY
                    };
                }
            }
        }
        let changes = ReplicaChanges {
            states: vec![StateEntry {
                id: BlockStateId(7),
                block: key("base:stone"),
                values: Vec::new(),
            }],
            sections: vec![SnapshotSection {
                pos: SectionPos::ZERO,
                cells: SnapshotCells::Dense(cells),
            }],
            removed_sections: Vec::new(),
            entities: Vec::new(),
            removed_entities: Vec::new(),
        };
        scene.apply_changes(&changes).expect("apply");
        assert!(!query(&scene, BlockPos::new(3, 0, 4)));
        assert!(query(&scene, BlockPos::new(5, 0, 5)));
        // The replaced section rebuilds despite matching revision numbers,
        // and the neighbour rebuilds because it observed the old revision.
        assert_eq!(scene.refresh_meshes(), 2);
    }

    #[test]
    fn walker_spawns_mid_air_walks_and_ends_standing() {
        let mut scene = Scene::new(3).expect("scene");
        scene
            .spawn_walker(Position::new(8.5, 13.3, 8.5))
            .expect("spawn");
        let mut final_position = Position::new(8.5, 13.3, 8.5);
        let mut grounded = false;
        for _ in 0..31 {
            let (position, landed) = scene
                .step_walker([0.0, 0.0, -1.0], false, 0.05)
                .expect("step");
            final_position = position;
            grounded = landed;
        }
        // Walked into negative z and ended standing on the terrain.
        assert!(final_position.z < 8.0, "moved to {final_position:?}");
        assert!(grounded);
        assert!(final_position.y > 0.0 && final_position.y < 16.0);
    }

    #[test]
    fn walker_falls_until_it_lands_and_stays() {
        let mut scene = Scene::new(1).expect("scene builds");
        scene
            .spawn_walker(Position::new(8.5, 12.0, 8.5))
            .expect("spawn");
        let mut settled = Position::new(8.5, 12.0, 8.5);
        for _ in 0..100 {
            let (position, grounded) = scene
                .step_walker([0.0, 0.0, 0.0], false, 0.05)
                .expect("step");
            settled = position;
            if grounded {
                break;
            }
        }
        let (_, grounded) = scene
            .step_walker([0.0, 0.0, 0.0], false, 0.05)
            .expect("settle");
        assert!(grounded, "must land within five seconds");
        // Standing on the surface: inside the loaded layer, x and z unchanged.
        assert!((settled.x - 8.5).abs() < 1e-9);
        assert!((settled.z - 8.5).abs() < 1e-9);
        assert!(settled.y > 0.0 && settled.y < 16.0);
    }

    #[test]
    fn walker_walks_forward_over_rolling_terrain() {
        let mut scene = Scene::new(1).expect("scene builds");
        scene
            .spawn_walker(Position::new(8.5, 12.0, 8.5))
            .expect("spawn");
        // Warm up until grounded.
        for _ in 0..100 {
            let (_, grounded) = scene
                .step_walker([0.0, 0.0, 0.0], false, 0.05)
                .expect("step");
            if grounded {
                break;
            }
        }
        let start = scene.walker_position().expect("walker");
        let mut final_position = start;
        for _ in 0..60 {
            final_position = scene
                .step_walker([0.0, 0.0, -1.0], false, 0.05)
                .expect("step")
                .0;
        }
        // Three seconds of walking moves along the surface until a terrain
        // step blocks the collider; hills stop it early, so only require real
        // progress plus a plausible standing height.
        assert!(
            start.z - final_position.z > 2.0,
            "moved {:?} to {:?}",
            start,
            final_position
        );
        assert!(final_position.y > 0.0 && final_position.y < 16.0);
        assert!((final_position.x - start.x).abs() < 1e-9);
    }

    #[test]
    fn walker_jumps_and_lands_back() {
        let mut scene = Scene::new(1).expect("scene builds");
        scene
            .spawn_walker(Position::new(8.5, 12.0, 8.5))
            .expect("spawn");
        for _ in 0..100 {
            let (_, grounded) = scene
                .step_walker([0.0, 0.0, 0.0], false, 0.05)
                .expect("step");
            if grounded {
                break;
            }
        }
        let ground = scene.walker_position().expect("walker").y;

        let (risen, grounded) = scene
            .step_walker([0.0, 0.0, 0.0], true, 0.05)
            .expect("jump tick");
        assert!(!grounded);
        assert!(risen.y > ground, "first jump tick must rise");

        let mut landed = risen;
        let mut on_ground = false;
        for _ in 0..100 {
            let (position, grounded) = scene
                .step_walker([0.0, 0.0, 0.0], false, 0.05)
                .expect("fall");
            landed = position;
            on_ground = grounded;
            if grounded {
                break;
            }
        }
        assert!(on_ground, "must land again");
        assert!((landed.y - ground).abs() < 0.5, "back at {landed:?}");
    }
}
