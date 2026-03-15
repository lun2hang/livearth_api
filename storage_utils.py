from google.cloud import storage
import os

# 设置环境变量指向你的密钥文件 (确保该文件存在，并强烈建议将其加入 .gitignore 以防泄露)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "livearth-251221-30bdbf1e98b3.json"

class GCSStorage:
    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    async def upload_file(self, file_content: bytes, destination_blob_name: str, content_type: str) -> str:
        """
        上传字节流到 GCS
        """
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_string(file_content, content_type=content_type)
        
        # 返回公网访问地址
        return f"https://storage.googleapis.com/{self.bucket.name}/{destination_blob_name}"

# 初始化单例 (请确认 "vision-uber-assets" 是你在 GCS 控制台创建的真实的 Bucket 名称)
gcs_service = GCSStorage(bucket_name="livearth-assets")