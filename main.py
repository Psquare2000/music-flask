from flask import Flask, jsonify
import os

app = Flask(__name__)


@app.route('/predict/', methods=['GET'])
def respond():
    name1=request.args.get("name1",None)
    year1=request.args.get("year1",None)
    name2=request.args.get("name2",None)
    year2=request.args.get("year2",None)
    name3=request.args.get("name3",None)
    year3=request.args.get("year3",None)
    name4=request.args.get("name4",None)
    year4=request.args.get("year4",None)
    name5=request.args.get("name5",None)
    year5=request.args.get("year5",None)

    response = {}

    # Check if the user sent a name at all
    if not name1:
        response["ERROR"] = "No name found. Please send a name."
    # # Check if the user entered a number
    # elif str(name).isdigit():
    #     response["ERROR"] = "The name can't be numeric. Please send a string."
    else:

      response["MESSAGE"] = recommend_songs([{'name': name1, 'year':int(year1)},
                {'name': name2, 'year': int(year2)},
                {'name': name3, 'year': int(year3)},
                {'name': name4, 'year': int(year4)},
                {'name': name5, 'year': int(year5)}],  data_songs)


    Return the response in json format
    response = "hi"
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
