from roboflow import Roboflow
rf = Roboflow(api_key="SHqXH9Hthjol3EotLlHd")
project = rf.workspace("ctarg").project("license_plate-naqg1")
dataset = project.version(5).download("voc")