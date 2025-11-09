## Folder Structure

Here’s how we’ll refactor your single-file app:

```bash
face_recognition_service/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   └── routes_face.py          # All face endpoints
│   │   └── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                    # App settings and env vars
│   │   └── logging_config.py            # Logging setup
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py                   # Pydantic response/request models
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── face_service.py              # Face recognition logic
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── image_utils.py               # Image loading, preprocessing
│   │
│   ├── main.py                          # App entrypoint (creates FastAPI instance)
│   └── __init__.py
│
├── requirements.txt
├── .env                                 # (optional) configs like DEBUG=True
├── .gitignore
└── README.md
```
