# 基础游戏界面素材

ui目录保存面板、按钮三态、槽位、进度条、准星和关闭图标，icons/items保存石头、玻璃半砖与植物图标。PNG使用原始像素和透明通道，程序通过整数倍缩放显示；素材原文件不因窗口尺寸改变。

| 素材 | 原始尺寸 | 九宫格边距 |
|---|---|---|
| panel_window | 24x24 | 4 |
| panel_tooltip | 16x16 | 3 |
| button_normal、button_hover、button_pressed | 24x20 | 4 |
| slot_normal、slot_selected | 24x24 | 不拉伸 |
| progress_bg | 64x8 | 不拉伸 |
| progress_fill | 60x4 | 按进度裁切，显示时着色 |
| crosshair | 9x9 | 不拉伸 |
| icon_close | 12x12 | 不拉伸 |
| icons/items内图标 | 16x16 | 不拉伸 |

透明面板中心由UI主题提供底色。进度填充保留原图明暗并乘以主题色，素材文件保持原样。locale资源采用每行key=value，zh-Hans与en的键对应；用户界面文案不写入Rust源码。

客户端编译时嵌入PNG与文案，修改素材后需要重新编译。字体由运行环境提供，不复制或分发操作系统字体。可用TORSE_UI_FONT指定字体文件；系统字体不含中文字形时选择英文文案。
