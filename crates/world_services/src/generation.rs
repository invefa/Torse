//! Deterministic terrain generation for the M1 prototype.
//!
//! Generation must depend only on the world seed, the target coordinates, and
//! the rules version. It must not read already generated neighbors, and it must
//! not depend on which worker runs first. Both generators here are pure
//! functions of their inputs, so the same section always receives identical
//! cells no matter how the caller schedules the work.
//!
//! The height field uses integer lattice noise with bilinear interpolation. A
//! float implementation would also be deterministic on one platform, but integer
//! arithmetic keeps the seam between sections independent of rounding mode and
//! makes the determinism test meaningful across machines.

use torse_types::SectionPos;
use torse_world::{BlockStateId, Section, SectionError, WorldBounds};

/// Fixed parameters shared by the built-in generators.
#[derive(Clone, Copy, Debug)]
pub struct GenerationParams {
    /// Section side length owned by the generator.
    pub side: u32,
    /// Supported vertical extent.
    pub bounds: WorldBounds,
    /// Block used for generated ground.
    pub ground: BlockStateId,
    /// Block used for open space.
    pub air: BlockStateId,
    /// Rules version recorded by the world and carried into regeneration.
    pub generation_version: u32,
}

/// Pluggable terrain source.
///
/// The core does not know how terrain is produced. An application assembles a
/// generator through this interface, so alternative algorithms can be swapped
/// without touching storage, lifecycle, or simulation code. The generator must
/// be a pure function of the world coordinate, the seed it captured, and
/// [`GenerationParams::generation_version`].
pub trait SectionGenerator {
    /// Section side length the generator produces.
    fn side(&self) -> u32;

    /// Rules version recorded with the generated terrain.
    fn generation_version(&self) -> u32;

    /// Produce one section. The result must not depend on generation order.
    fn generate(&self, section: SectionPos) -> Result<Section, SectionError>;
}

impl GenerationParams {
    /// Validate the shared parameters once instead of inside every section.
    pub fn new(
        side: u32,
        bounds: WorldBounds,
        ground: BlockStateId,
        air: BlockStateId,
    ) -> Result<Self, SectionError> {
        let _ = Section::empty(side)?;
        Ok(Self {
            side,
            bounds,
            ground,
            air,
            generation_version: 1,
        })
    }

    /// Set the rules version recorded with the terrain.
    pub const fn with_version(mut self, version: u32) -> Self {
        self.generation_version = version;
        self
    }
}

/// Trivial generator that fills everything below a flat surface.
#[derive(Clone, Copy, Debug)]
pub struct FlatTerrain {
    params: GenerationParams,
    surface_y: i64,
}

impl FlatTerrain {
    /// Create a flat generator.
    pub fn new(params: GenerationParams, surface_y: i64) -> Self {
        Self { params, surface_y }
    }

    /// Height of the flat surface.
    pub const fn surface_y(&self) -> i64 {
        self.surface_y
    }

    /// Height at any horizontal coordinate.
    pub const fn height_at(&self, _x: i64, _z: i64) -> i64 {
        self.surface_y
    }

    /// Generate one section.
    pub fn generate(&self, section: SectionPos) -> Result<Section, SectionError> {
        fill_section(&self.params, section, |x, z| self.height_at(x, z))
    }
}

impl SectionGenerator for FlatTerrain {
    fn side(&self) -> u32 {
        self.params.side
    }

    fn generation_version(&self) -> u32 {
        self.params.generation_version
    }

    fn generate(&self, section: SectionPos) -> Result<Section, SectionError> {
        FlatTerrain::generate(self, section)
    }
}

/// Rolling-hills generator driven by a 64-bit seed.
#[derive(Clone, Copy, Debug)]
pub struct HillTerrain {
    params: GenerationParams,
    seed: u64,
    base_y: i64,
    amplitude: i64,
}

impl HillTerrain {
    /// Create a seeded generator.
    ///
    /// `amplitude` is the maximum vertical deviation from `base_y` in blocks.
    pub fn new(params: GenerationParams, seed: u64, base_y: i64, amplitude: i64) -> Self {
        Self {
            params,
            seed,
            base_y,
            amplitude: amplitude.max(0),
        }
    }

    /// Seed used by this generator.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Ground height at a horizontal coordinate.
    ///
    /// The coarse lattice controls the overall shape and the fine lattice adds
    /// local variation. Both are sampled at lattice corners and interpolated, so
    /// adjacent sections share exactly the same seam values.
    pub fn height_at(&self, x: i64, z: i64) -> i64 {
        let coarse = value_noise(self.seed ^ 0x51ED_2701, x, z, 16);
        let fine = value_noise(self.seed ^ 0x2545_F491, x, z, 4);
        let mixed = (coarse * 3 + fine) / 4;
        let deviation = (mixed - 128) * self.amplitude / 128;
        self.base_y + deviation
    }

    /// Generate one section.
    pub fn generate(&self, section: SectionPos) -> Result<Section, SectionError> {
        fill_section(&self.params, section, |x, z| self.height_at(x, z))
    }
}

impl SectionGenerator for HillTerrain {
    fn side(&self) -> u32 {
        self.params.side
    }

    fn generation_version(&self) -> u32 {
        self.params.generation_version
    }

    fn generate(&self, section: SectionPos) -> Result<Section, SectionError> {
        HillTerrain::generate(self, section)
    }
}

fn fill_section(
    params: &GenerationParams,
    section: SectionPos,
    height_at: impl Fn(i64, i64) -> i64,
) -> Result<Section, SectionError> {
    let mut result = Section::new(params.side, params.air)?;
    let side = params.side;
    let origin = section.to_block(torse_types::LocalPos { x: 0, y: 0, z: 0 }, side)?;
    // Validate both ends before visiting cells. This protects the additions
    // below as well as the section-to-block multiplications.
    section.to_block(
        torse_types::LocalPos {
            x: side - 1,
            y: side - 1,
            z: side - 1,
        },
        side,
    )?;
    let (base_x, base_y, base_z) = (origin.x, origin.y, origin.z);

    // Column heights are constant down the section, so evaluate the height
    // function once per (x, z) column instead of once per cell; the noise
    // path is the dominant generation cost otherwise.
    let mut heights = vec![0i64; (side as usize) * (side as usize)];
    for local_z in 0..side {
        let world_z = base_z + i64::from(local_z);
        for local_x in 0..side {
            let world_x = base_x + i64::from(local_x);
            heights[(local_z as usize) * (side as usize) + local_x as usize] =
                height_at(world_x, world_z);
        }
    }
    for local_y in 0..side {
        let world_y = base_y + i64::from(local_y);
        if !params.bounds.contains_y(world_y) {
            continue;
        }
        for local_z in 0..side {
            for local_x in 0..side {
                if world_y < heights[(local_z as usize) * (side as usize) + local_x as usize] {
                    result.set(
                        torse_types::LocalPos {
                            x: local_x,
                            y: local_y,
                            z: local_z,
                        },
                        params.ground,
                    )?;
                }
            }
        }
    }
    Ok(result)
}

fn lattice(seed: u64, x: i64, z: i64) -> i64 {
    (hash2(seed, x, z) % 256) as i64
}

fn value_noise(seed: u64, x: i64, z: i64, cell: i64) -> i64 {
    let gx = x.div_euclid(cell);
    let gz = z.div_euclid(cell);
    let fx = x.rem_euclid(cell);
    let fz = z.rem_euclid(cell);
    let wx = fx * 256 / cell;
    let wz = fz * 256 / cell;

    let c00 = lattice(seed, gx, gz);
    let c10 = lattice(seed, gx + 1, gz);
    let c01 = lattice(seed, gx, gz + 1);
    let c11 = lattice(seed, gx + 1, gz + 1);

    let near = c00 * (256 - wx) + c10 * wx;
    let far = c01 * (256 - wx) + c11 * wx;
    (near * (256 - wz) + far * wz) / 65_536
}

/// Stable 64-bit hash of a seed and two coordinates.
///
/// The constants are the SplitMix64 finalizer; the function intentionally uses
/// only wrapping arithmetic so the value is identical on every target.
fn hash2(seed: u64, x: i64, z: i64) -> u64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value ^= (x as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    value = mix(value);
    value ^= (z as u64).wrapping_mul(0xA24B_AED4_963E_E407);
    mix(value)
}

fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> GenerationParams {
        GenerationParams::new(
            16,
            WorldBounds::new(-64, 128).expect("bounds are valid"),
            BlockStateId(1),
            BlockStateId::EMPTY,
        )
        .expect("side is valid")
    }

    #[test]
    fn generators_are_interchangeable_behind_the_trait() {
        let params = params().with_version(3);
        let generators: Vec<Box<dyn SectionGenerator>> = vec![
            Box::new(FlatTerrain::new(params, 4)),
            Box::new(HillTerrain::new(params, 9, 4, 3)),
        ];
        for generator in &generators {
            assert_eq!(generator.side(), 16);
            assert_eq!(generator.generation_version(), 3);
            let section = generator
                .generate(SectionPos::ZERO)
                .expect("generation succeeds");
            assert_eq!(section.side(), 16);
        }
    }

    #[test]
    fn flat_generator_fills_only_below_the_surface() {
        let generator = FlatTerrain::new(params(), 8);
        let section = generator
            .generate(SectionPos::ZERO)
            .expect("generation succeeds");
        assert_eq!(
            section.get(torse_types::LocalPos { x: 3, y: 3, z: 3 }),
            Ok(BlockStateId(1))
        );
        assert_eq!(
            section.get(torse_types::LocalPos { x: 3, y: 9, z: 3 }),
            Ok(BlockStateId::EMPTY)
        );
    }

    #[test]
    fn hills_are_deterministic_for_the_same_seed() {
        let first = HillTerrain::new(params(), 42, 4, 6);
        let second = HillTerrain::new(params(), 42, 4, 6);
        for x in [-37, -1, 0, 1, 37] {
            for z in [-37, -1, 0, 1, 37] {
                assert_eq!(first.height_at(x, z), second.height_at(x, z));
            }
        }
        let different = HillTerrain::new(params(), 43, 4, 6);
        let changes = (0..32)
            .filter(|x| first.height_at(*x, 0) != different.height_at(*x, 0))
            .count();
        assert!(changes > 0, "a different seed must change the terrain");
    }

    #[test]
    fn heights_stay_within_the_configured_amplitude() {
        let generator = HillTerrain::new(params(), 7, 0, 5);
        for x in -200..200 {
            let height = generator.height_at(x, x / 3);
            assert!(
                (-5..=5).contains(&height),
                "height {height} left the amplitude"
            );
        }
    }

    #[test]
    fn every_cell_follows_the_world_height() {
        // The fill must dispatch on the world coordinate, not on the local one.
        // Reading the world height back for sampled cells across a section
        // boundary catches the classic "local x used as terrain input" bug.
        let generator = HillTerrain::new(params(), 99, 4, 8);
        for section in [
            SectionPos::new(-1, 0, -1),
            SectionPos::ZERO,
            SectionPos::new(1, 0, 1),
        ] {
            let generated = generator.generate(section).expect("generation succeeds");
            for (lx, ly, lz) in [(0u32, 3u32, 0u32), (15, 7, 5), (7, 12, 15)] {
                let world_x = section.x * 16 + i64::from(lx);
                let world_y = section.y * 16 + i64::from(ly);
                let world_z = section.z * 16 + i64::from(lz);
                let expected = if world_y < generator.height_at(world_x, world_z) {
                    BlockStateId(1)
                } else {
                    BlockStateId::EMPTY
                };
                let actual = generated
                    .get(torse_types::LocalPos {
                        x: lx,
                        y: ly,
                        z: lz,
                    })
                    .expect("in range");
                assert_eq!(actual, expected, "cell ({lx}, {ly}, {lz}) in {section:?}");
            }
        }
    }

    #[test]
    fn cells_outside_the_world_bounds_stay_empty() {
        let params = GenerationParams::new(
            16,
            WorldBounds::new(0, 32).expect("bounds are valid"),
            BlockStateId(1),
            BlockStateId::EMPTY,
        )
        .expect("side is valid");
        let generator = FlatTerrain::new(params, 8);
        let high = generator
            .generate(SectionPos::new(0, 4, 0))
            .expect("generation succeeds");
        assert_eq!(
            high.allocated_cells(),
            0,
            "outside the world it writes nothing"
        );
    }
    #[test]
    fn unrepresentable_section_coordinates_are_rejected_without_panicking() {
        let flat = FlatTerrain::new(params(), 8);
        let hills = HillTerrain::new(params(), 7, 7, 9);
        for section in [
            SectionPos::new(i64::MAX, 0, 0),
            SectionPos::new(i64::MIN, 0, 0),
            SectionPos::new(0, i64::MAX, 0),
            SectionPos::new(0, 0, i64::MIN),
        ] {
            assert!(flat.generate(section).is_err());
            assert!(hills.generate(section).is_err());
        }
    }

    #[test]
    fn representable_sections_at_coordinate_limits_still_generate() {
        let generator = HillTerrain::new(params(), 7, 7, 9);
        for x in [i64::MIN / 16, i64::MAX / 16] {
            for z in [i64::MIN / 16, i64::MAX / 16] {
                assert_eq!(
                    generator.generate(SectionPos::new(x, 0, z)).unwrap().side(),
                    16
                );
            }
        }
    }
}
