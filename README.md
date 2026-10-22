# Blender UI

A React-based frontend user interface and a FastAPI backend with PostgreSQL session tracking to manage and process user prompts.

## Requirements

* Python 3.10+
* Node.js v18+ & npm
* PostgreSQL database server

## Setup

### Clone Repository

```bash
git clone https://github.com/sarran99/blender-script.git
cd blender-script

```

### Create a virtual environment:
```bash
python -m venv venv

```

### Activate the virtual environment:
* **Windows:**
```bash
venv\Scripts\activate

```


* **Linux/Mac:**
```bash
source venv/bin/activate

```


### Install dependencies:
```bash
pip install -r requirements.txt

```

### Backend Setup (FastAPI)

1. Navigate to the backend directory:
```bash
cd backend

```


2. Run the API:
```bash
uvicorn main:app --reload

```


*If successful, the backend server will be running on `http://127.0.0.1:8000*`

### Frontend Setup (React)

1. Open a new terminal window, navigate to the frontend directory, and install dependencies:
```bash
cd frontend
npm install

```


2. Run the frontend application:
```bash
npm start

```


*If successful, your browser will open the user interface at `http://127.0.0.1:3000*`

### Database Configuration

1. Ensure your local PostgreSQL server instance is running.
2. Create a new database named `blender`.
3. Verify or update the database credentials/connection string inside your backend configurations (`database.py` or `.env` file) to match your local setup.

---

## Notes

* **CORS Configuration:** Ensure the FastAPI application allows cross-origin requests from the React development server port (`3000`) for seamless data communication.
