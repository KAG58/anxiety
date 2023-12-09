from flask import Flask ,render_template,request
import numpy as np
import pickle
with open('stress.pkl','rb') as m:
    model = pickle.load(m)

with open('scaler.pkl','rb') as s:
    scaler = pickle.load(s)
#create an object instance
app=Flask(__name__)


@app.route('/',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        anxiety_level=int(request.form['anxiety_level'])
        self_esteem=int(request.form['self_esteem'])
        mental_health_history=int(request.form['mental_health_history'])
        depression=int(request.form['depression'])
        headache=int(request.form['headache'])
        blood_pressure=int(request.form['blood_pressure'])
        sleep_quality=int(request.form['sleep_quality'])
        breathing_problem=int(request.form['breathing_problem'])
        noise_level=int(request.form['noise_level'])
        living_conditions=int(request.form['living_conditions'])
        safety=int(request.form['safety'])
        basic_needs=int(request.form['basic_needs'])
        academic_performance=int(request.form['academic_performance'])
        study_load=int(request.form['study_load'])
        teacher_student_relationship=int(request.form['teacher_student_relationship'])
        future_career_concerns=int(request.form['future_career_concerns'])
        social_support=int(request.form['social_support'])
        peer_pressure=int(request.form['peer_pressure'])
        extracurricular_activities=int(request.form['extracurricular_activities'])
        bullying=int(request.form['bullying'])
        user_input = np.array([[anxiety_level,self_esteem,mental_health_history
                                ,depression,headache,blood_pressure,sleep_quality,
                                breathing_problem,noise_level,living_conditions,safety,
                                basic_needs,academic_performance,study_load,
                                teacher_student_relationship,future_career_concerns,
                                social_support,peer_pressure,extracurricular_activities,
                                bullying]])
        user_input_scaled = scaler.transform(user_input)
        pred = model.predict(user_input_scaled)
        if pred[0]==1:
            prediction='Normal'
        elif pred[0]==1:
            prediction='Not Bad'
        else:
            prediction='Serious'
        return render_template("index.html",prediction=prediction)       
    return render_template("index.html")


if __name__=="__main__":
    app.run(use_reloader=True,debug=True)
                              
