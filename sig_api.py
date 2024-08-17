from fastapi import FastAPI
import uvicorn
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(debug=True)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get('/predictions')
async def prediction(
    Gender: bool, StudyTimeWeekly: float, Absences: int, Tutoring: bool, ParentalSupport: int,
    Extracurricular: bool, Sports: bool, Music: bool, GradeClass: int
):
    load_model = pickle.load(open('C:/Users/Nceba.Gagaza/OneDrive - MRI Software/Documents/101 models/Student_performance/regression_model_api.pkl', 'rb'))
    make_predictions = load_model.predict(
        [[
            Gender, StudyTimeWeekly, Absences, Tutoring, ParentalSupport,
            Extracurricular, Sports, Music, GradeClass
        ]]
    )

    output_value = round(make_predictions[0],2)
    return {"Learner predicted GPA Score: {}".format(output_value)}

if __name__ == '_main_':
    uvicorn.run(app)

# uvicorn sig_api:app --host 127.0.0.1 --port 8001
