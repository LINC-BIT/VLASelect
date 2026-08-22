import base64
import pickle


def serialize_tensor(tensor):
    """将PyTorch Tensor序列化为字符串"""
    return base64.b64encode(pickle.dumps(tensor)).decode('utf-8')

def deserialize_tensor(s):
    """将字符串反序列化为PyTorch Tensor"""
    return pickle.loads(base64.b64decode(s.encode('utf-8')))


