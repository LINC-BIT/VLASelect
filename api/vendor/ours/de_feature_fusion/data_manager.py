# broker.py
import uvicorn
import pickle
import base64
from fastapi import FastAPI
from threading import Lock


app = FastAPI()

clients_info_db = {}
# 使用线程安全的字典来存储每个客户端最新的特征
# 键: client_id, 值: base64编码的序列化tensor
features_db = {}
metrics_db = {}

clients_info_db_lock = Lock()
features_db_lock = Lock()
metrics_db_lock = Lock()


from pydantic import BaseModel as _BaseModel

class RegisterClientReq(_BaseModel):
    name: str
    local_feature_dim: int
    local_action_dim: int = 0

@app.post("/register_client")
async def register_client(client_info: RegisterClientReq):
    """客户端调用此接口来注册自己的信息"""
    with clients_info_db_lock:
        clients_info_db[client_info.name] = {
            'local_feature_dim': client_info.local_feature_dim,
            'local_action_dim': client_info.local_action_dim,
        }
    return {"message": f"Client {client_info.name} registered."}


@app.get("/clients_info")
async def get_clients_info():
    """客户端调用此接口来获取所有已注册的客户端信息"""
    with clients_info_db_lock:
        return clients_info_db


from pydantic import BaseModel

class FeatureReq(BaseModel):
    client_id: str
    feature: str

@app.post("/upload_feature")
async def upload_feature(req: FeatureReq):
    """客户端调用此接口来更新自己的特征"""
    with features_db_lock:
        # print(client_id, feature[:30], flush=True) # 打印部分特征信息以观察更新情况
        features_db[req.client_id] = req.feature
    return {"message": f"Feature for client {req.client_id} updated."}


@app.get("/get_feature")
async def get_feature(client_id: str):
    """客户端调用此接口来获取其它客户端的特征"""
    with features_db_lock:
        feature = features_db.get(client_id, None)
    return {"feature": feature}


class MetricsReq(BaseModel):
    client_id: str
    time: float
    metrics: dict

@app.post("/report_metrics")
async def report_metrics(req: MetricsReq):
    """客户端调用此接口来更新自己的特征"""
    with metrics_db_lock:
        # print(client_id, feature[:30], flush=True) # 打印部分特征信息以观察更新情况
        if req.client_id not in metrics_db:
            metrics_db[req.client_id] = []
        metrics_db[req.client_id] += [{'time': req.time, 'metrics': req.metrics}]
    return {"message": f"Feature for client {req.client_id} updated."}


if __name__ == "__main__":
    # 运行: uvicorn broker:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
