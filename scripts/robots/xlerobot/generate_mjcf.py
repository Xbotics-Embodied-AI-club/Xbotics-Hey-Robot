"""从上游 XLeRobot 资产生成 MuJoCo MJCF 场景文件（xlerobot.xml）。

用途：
  - 这是开发期工具，普通用户通常不需要跑。
  - 当上游 XLeRobot 仓库的 URDF / 网格 / 相机参数更新后，用这个脚本重新生成
    assets/robots/xlerobot/xlerobot.xml 和 xlerobot.official.generated.urdf。
  - 生成的文件被仿真驱动 (XLeRobotSimDriver) 加载用于训练和回放。

常见用法：
  # 默认从本地路径 D:\\agent_robot\\XLeRobot 读取上游资产并生成
  uv run python scripts/robots/xlerobot/generate_mjcf.py

  # 跑完后会输出：generated <repo>/assets/robots/xlerobot/xlerobot.xml

输出说明：
  - 生成的 MJCF 路径会打印在最后一行。
  - 失败时通常是因为上游仓库路径不存在或网格文件缺失。

退出码：
  - 0：生成成功
  - 非 0：异常（看堆栈）

注意：
  本脚本依赖硬编码的上游仓库路径 OFFICIAL_ASSET_ROOT 和 OFFICIAL_MUJOCO_ROOT，
  如需在不同机器上运行请先修改这些常量。
"""

from __future__ import annotations

import re
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ASSET_ROOT = Path(
    r"D:\agent_robot\XLeRobot\simulation\Maniskill\assets\xlerobot"
)
OFFICIAL_MUJOCO_ROOT = Path(r"D:\agent_robot\XLeRobot\simulation\mujoco")
OFFICIAL_MUJOCO_ASSET_ROOT = OFFICIAL_MUJOCO_ROOT / "assets"
ROBOT_ROOT = REPO_ROOT / "assets" / "robots" / "xlerobot"
MESH_ROOT = ROBOT_ROOT / "meshes"
GENERATED_URDF = ROBOT_ROOT / "xlerobot.official.generated.urdf"
GENERATED_MJCF = ROBOT_ROOT / "xlerobot.xml"

EPS_MASS = "0.001"
EPS_INERTIA = "1e-6"

LEKIWI_MESH_NAMES = {
    "drive_motor_mount-v4-3",
    "base_plate_layer1-v5-1",
    "ST3215_Servo_Motor-v1-2",
    "3_8-Hex-Bore-Plastic-VersaHub-217-5828-v2-2",
    "4-Omni-Directional-Wheel_Single_Body-v1-2",
    "drive_motor_mount-v4-2",
    "ST3215_Servo_Motor-v1-1",
    "3_8-Hex-Bore-Plastic-VersaHub-217-5828-v2-1",
    "4-Omni-Directional-Wheel_Single_Body-v1-1",
    "drive_motor_mount-v4-1",
    "ST3215_Servo_Motor-v1",
    "3_8-Hex-Bore-Plastic-VersaHub-217-5828-v2",
    "4-Omni-Directional-Wheel_Single_Body-v1",
    "servo_controller_mount-v1",
    "lipo_battery_mount-v1",
    "Battery---Battery-5.2-Ah-DC5521-Plug-v2",
    "94868A713_NO-THREADS_Female-Threaded-Hex-Standoff",
    "base_plate_layer2-v1",
}


def _convert_binary_ply_to_stl(ply_path: Path, stl_path: Path) -> None:
    data = ply_path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"unsupported PLY format: {ply_path}")

    vertex_count = int(re.search(r"element vertex (\d+)", header).group(1))
    face_count = int(re.search(r"element face (\d+)", header).group(1))
    vertex_property_count = len(
        re.findall(r"property float ", header.split("element face", maxsplit=1)[0])
    )
    vertex_stride = 4 * vertex_property_count

    offset = header_end
    vertices: list[tuple[float, float, float]] = []
    for _ in range(vertex_count):
        vertices.append(struct.unpack_from("<fff", data, offset))
        offset += vertex_stride

    triangles: list[list[int]] = []
    for _ in range(face_count):
        index_count = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        indices = list(struct.unpack_from("<" + "I" * index_count, data, offset))
        offset += 4 * index_count
        triangles.extend(
            [indices[0], indices[idx], indices[idx + 1]]
            for idx in range(1, index_count - 1)
        )

    with stl_path.open("wb") as out:
        out.write(b"converted from official XLeRobot PLY".ljust(80, b" "))
        out.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            out.write(struct.pack("<fff", 0.0, 0.0, 0.0))
            for vertex_index in triangle:
                out.write(struct.pack("<fff", *vertices[vertex_index]))
            out.write(struct.pack("<H", 0))


def _official_mesh_refs() -> set[str]:
    tree = ET.parse(OFFICIAL_ASSET_ROOT / "xlerobot.urdf")
    refs: set[str] = set()
    for mesh in tree.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            refs.add(filename.replace("\\", "/"))
    return refs


def _copy_and_normalize_meshes(mesh_refs: set[str]) -> dict[str, str]:
    MESH_ROOT.mkdir(parents=True, exist_ok=True)
    replacements: dict[str, str] = {}
    keep: set[Path] = set()

    for ref in sorted(mesh_refs):
        if not ref.startswith("meshes/"):
            raise ValueError(f"unsupported mesh path in official URDF: {ref}")
        source = OFFICIAL_ASSET_ROOT / ref
        if not source.is_file():
            raise FileNotFoundError(source)
        target = ROBOT_ROOT / ref
        shutil.copy2(source, target)
        keep.add(target.resolve())
        if source.suffix.lower() == ".ply":
            converted = target.with_suffix(target.suffix + ".stl")
            _convert_binary_ply_to_stl(target, converted)
            replacements[f"meshes/{source.name}"] = f"meshes/{converted.name}"
            keep.add(converted.resolve())

    official_mujoco = ET.parse(OFFICIAL_MUJOCO_ROOT / "xlerobot.xml").getroot()
    for mesh in official_mujoco.findall("./asset/mesh"):
        name = mesh.attrib.get("name")
        filename = mesh.attrib.get("file")
        if name not in LEKIWI_MESH_NAMES or not filename:
            continue
        source = OFFICIAL_MUJOCO_ASSET_ROOT / filename
        target = MESH_ROOT / filename
        shutil.copy2(source, target)
        keep.add(target.resolve())

    for stale in MESH_ROOT.iterdir():
        if stale.is_file() and stale.resolve() not in keep:
            stale.unlink()

    return replacements


def _ensure_inertial(link: ET.Element) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", {"value": EPS_MASS})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": EPS_INERTIA,
                "ixy": "0",
                "ixz": "0",
                "iyy": EPS_INERTIA,
                "iyz": "0",
                "izz": EPS_INERTIA,
            },
        )
        return

    mass = inertial.find("mass")
    if mass is not None and float(mass.attrib.get("value", "0")) <= 0.0:
        mass.attrib["value"] = EPS_MASS

    inertia = inertial.find("inertia")
    if inertia is not None:
        for attr in ("ixx", "iyy", "izz"):
            if float(inertia.attrib.get(attr, "0")) <= 0.0:
                inertia.attrib[attr] = EPS_INERTIA


def _prepare_urdf(replacements: dict[str, str]) -> None:
    source_urdf = OFFICIAL_ASSET_ROOT / "xlerobot.urdf"
    text = source_urdf.read_text(encoding="utf-8")
    for before, after in replacements.items():
        text = text.replace(before, after)

    GENERATED_URDF.write_text(text, encoding="utf-8")
    tree = ET.parse(GENERATED_URDF)
    root = tree.getroot()

    mujoco_node = ET.Element("mujoco")
    mujoco_node.append(
        ET.Element(
            "compiler",
            {
                "balanceinertia": "true",
                "discardvisual": "false",
                "fusestatic": "false",
            },
        )
    )
    root.insert(0, mujoco_node)

    for link in root.findall("link"):
        _ensure_inertial(link)

    tree.write(GENERATED_URDF, encoding="utf-8", xml_declaration=True)


def _add_actuators(root: ET.Element) -> None:
    existing = root.find("actuator")
    if existing is not None:
        root.remove(existing)

    actuator = ET.SubElement(root, "actuator")
    for name, joint in (
        ("base_x", "root_x_axis_joint"),
        ("base_y", "root_y_axis_joint"),
        ("base_yaw", "root_z_rotation_joint"),
    ):
        ET.SubElement(actuator, "velocity", {"name": name, "joint": joint, "kv": "10"})

    for name, joint in (
        ("Rotation_R", "Rotation_2"),
        ("Pitch_R", "Pitch_2"),
        ("Elbow_R", "Elbow_2"),
        ("Wrist_Pitch_R", "Wrist_Pitch_2"),
        ("Wrist_Roll_R", "Wrist_Roll_2"),
        ("Jaw_R", "Jaw_2"),
        ("Rotation_L", "Rotation"),
        ("Pitch_L", "Pitch"),
        ("Elbow_L", "Elbow"),
        ("Wrist_Pitch_L", "Wrist_Pitch"),
        ("Wrist_Roll_L", "Wrist_Roll"),
        ("Jaw_L", "Jaw"),
    ):
        ET.SubElement(
            actuator,
            "position",
            {
                "name": name,
                "joint": joint,
                "kp": "50",
                "dampratio": "1",
                "forcerange": "-35 35",
                "ctrlrange": "-3.14158 3.14158",
            },
        )

    for name, joint in (
        ("wheel1", "root_x_axis_joint"),
        ("wheel2", "root_y_axis_joint"),
        ("wheel3", "root_z_rotation_joint"),
    ):
        ET.SubElement(
            actuator,
            "general",
            {
                "name": name,
                "joint": joint,
                "gainprm": "0",
                "biasprm": "0",
            },
        )

    for name, joint in (
        ("head_pan_hold", "head_pan_joint"),
        ("head_tilt_hold", "head_tilt_joint"),
    ):
        ET.SubElement(
            actuator,
            "position",
            {
                "name": name,
                "joint": joint,
                "kp": "25",
                "dampratio": "1",
                "forcerange": "-3 3",
                "ctrlrange": "-1.57 1.57",
            },
        )


def _is_lekiwi_mesh(name: str | None) -> bool:
    return name in LEKIWI_MESH_NAMES


def _copy_lekiwi_assets(root: ET.Element) -> None:
    source_root = ET.parse(OFFICIAL_MUJOCO_ROOT / "xlerobot.xml").getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    existing_meshes = {mesh.attrib.get("name") for mesh in asset.findall("mesh")}
    for mesh in source_root.findall("./asset/mesh"):
        name = mesh.attrib.get("name")
        filename = mesh.attrib.get("file")
        if not _is_lekiwi_mesh(name) or not filename or name in existing_meshes:
            continue
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": name,
                "content_type": "model/stl",
                "file": f"meshes/{filename}",
                "scale": mesh.attrib.get("scale", "1 1 1"),
            },
        )


def _strip_dynamic_nodes(body: ET.Element) -> None:
    for node in list(body):
        if node.tag in {"joint", "inertial"}:
            body.remove(node)
        elif node.tag == "geom":
            node.attrib["contype"] = "0"
            node.attrib["conaffinity"] = "0"
            node.attrib["density"] = "0"
            node.attrib["group"] = "1"
            node.attrib.pop("friction", None)
        elif node.tag == "body":
            _strip_dynamic_nodes(node)


def _add_lekiwi_chassis_visual(root: ET.Element) -> None:
    base_link = root.find(".//body[@name='base_link']")
    if (
        base_link is None
        or base_link.find("body[@name='lekiwi_chassis_visual']") is not None
    ):
        return

    source_root = ET.parse(OFFICIAL_MUJOCO_ROOT / "xlerobot.xml").getroot()
    chassis = source_root.find("./worldbody/body[@name='chassis']")
    if chassis is None:
        raise ValueError("official MuJoCo chassis body not found")

    visual = ET.fromstring(ET.tostring(chassis, encoding="unicode"))
    visual.attrib["name"] = "lekiwi_chassis_visual"
    for node in list(visual):
        if node.tag == "geom" or (
            node.tag == "body" and node.attrib.get("name") != "base_plate_layer1-v5-1"
        ):
            visual.remove(node)
    _strip_dynamic_nodes(visual)
    base_link.insert(0, visual)


def _hide_collision_geoms(root: ET.Element) -> None:
    for geom in root.findall(".//geom"):
        is_visual = (
            geom.attrib.get("group") == "1"
            and geom.attrib.get("density") == "0"
            and geom.attrib.get("contype") == "0"
            and geom.attrib.get("conaffinity") == "0"
        )
        if is_visual:
            continue

        geom.attrib["group"] = "3"
        geom.attrib["rgba"] = "0 0 0 0"


def _ensure_camera(
    body: ET.Element,
    *,
    name: str,
    quat: str,
    pos: str = "0 0 0",
    fovy: str,
) -> None:
    for camera in body.findall("camera"):
        if camera.attrib.get("name") == name:
            camera.attrib["pos"] = pos
            camera.attrib["quat"] = quat
            camera.attrib["fovy"] = fovy
            return

    ET.SubElement(
        body,
        "camera",
        {
            "name": name,
            "pos": pos,
            "quat": quat,
            "fovy": fovy,
        },
    )


def _add_official_cameras(root: ET.Element) -> None:
    # Official ManiSkill mounts cameras on these links with identity pose.
    # MuJoCo fixed cameras use a different camera frame convention, so the
    # +90deg X rotation aligns the native views with the official task-facing
    # camera direction. The small +Y offset moves the viewpoint to the lens
    # instead of leaving it inside the camera body mesh.
    mujoco_camera_quat = "0.7071068 0.7071068 0 0"
    lens_pos = "0 0.04 0"
    for body_name, camera_name, fovy in (
        ("head_camera_link", "front", "91.673"),
        ("Left_Arm_Camera", "left_wrist", "74.485"),
        ("Right_Arm_Camera", "right_wrist", "74.485"),
    ):
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(
                f"official camera body not found in generated MJCF: {body_name}"
            )
        _ensure_camera(
            body,
            name=camera_name,
            pos=lens_pos,
            quat=mujoco_camera_quat,
            fovy=fovy,
        )


def _write_generated_mjcf() -> None:
    model = mujoco.MjModel.from_xml_path(str(GENERATED_URDF))
    mujoco.mj_saveLastXML(str(GENERATED_MJCF), model)

    tree = ET.parse(GENERATED_MJCF)
    root = tree.getroot()
    _copy_lekiwi_assets(root)
    _add_lekiwi_chassis_visual(root)
    _hide_collision_geoms(root)
    _add_official_cameras(root)
    _add_actuators(root)
    tree.write(GENERATED_MJCF, encoding="utf-8", xml_declaration=False)


def main() -> None:
    mesh_refs = _official_mesh_refs()
    replacements = _copy_and_normalize_meshes(mesh_refs)
    _prepare_urdf(replacements)
    _write_generated_mjcf()
    print(f"generated {GENERATED_MJCF}")


if __name__ == "__main__":
    main()
