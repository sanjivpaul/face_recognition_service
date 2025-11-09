from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Facial Recognition Service"
    MODEL_VERSION: str = "face_recognition_v1"
    FACE_THRESHOLD: float = 0.6

    class Config:
        env_file = ".env"

settings = Settings()
