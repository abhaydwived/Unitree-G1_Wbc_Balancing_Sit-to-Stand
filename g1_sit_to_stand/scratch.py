import mujoco
import mink

model = mujoco.MjModel.from_xml_path("../robot/scene_sit_to_stand.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

configuration = mink.Configuration(model)
print("world name:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0))

try:
    t = configuration.get_transform("left_foot", "site", "world", "body")
    print("Success get_transform", t.translation())
except Exception as e:
    print("Error:", e)
