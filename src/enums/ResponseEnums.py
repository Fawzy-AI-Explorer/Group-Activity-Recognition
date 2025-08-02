from enum import Enum

class ResponseSignal(Enum):

    FILE_VALIDATE_SUCCESS = "File validation successful"
    FILE_TYPE_NOT_SUPPORTED = "File type not allowed"
    FILE_SIZE_EXCEEDS_LIMIT = "File size exceeds the maximum limit"
    FILE_UPLOADED_SUCCESSFULLY = "File uploaded successfully"
    FILE_UPLOADED_FAILED = "File upload failed"
