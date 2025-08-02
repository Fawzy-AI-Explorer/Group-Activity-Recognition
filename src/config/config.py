from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configuration settings for the application.
    """

    APP_NAME: str 
    APP_VERSION: str
    API: str

    FILE_ALLOWED_TYPES: list[str] 
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    class Config:
        env_file = r"E:\DATA SCIENCE\MINI-RAG-APP\min-rag\src\.env"

def get_settings():

    return Settings()

# if __name__ == "__main__":
#     settings = get_settings()
#     print(settings.APP_NAME)
#     print(settings.APP_VERSION)
#     print(settings.API)
#     print(settings.FILE_ALLOWED_TYPES)
#     print(settings.FILE_MAX_SIZE)


# min-rag/src/
# to run : python -m helpers.config