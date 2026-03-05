import yaml
import mujoco
from mujoco import viewer
import numpy as np
import os

# -------------------------------
# 1️⃣ 定义文件路径
# -------------------------------
# 假设 MJCF 文件在 mjcf/ 文件夹下
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "mjcf", "xarm.xml")
PARAMS_YAML_PATH = os.path.join(os.path.dirname(__file__), "params.yaml")

# -------------------------------
# 2️⃣ 加载 YAML 参数
# -------------------------------
with open(PARAMS_YAML_PATH, "r") as f:
    params = yaml.safe_load(f)

# -------------------------------
# 3️⃣ 加载 MJCF 模型
# -------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

# -------------------------------
# 4️⃣ 设置初始关节位置
# -------------------------------
for joint in params["arm"]["joints"]:
    joint_name = joint["name"]
    home_pos = joint["home_pos"]
    joint_id = model.joint_name2id(joint_name)
    data.qpos[joint_id] = home_pos

for joint in params["arm"]["gripper"]:
    joint_name = joint["name"]
    home_pos = joint["home_pos"]
    joint_id = model.joint_name2id(joint_name)
    data.qpos[joint_id] = home_pos

# -------------------------------
# 5️⃣ 可视化
# -------------------------------
viewer.launch(model, data)