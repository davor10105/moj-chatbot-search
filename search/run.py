from flask import Flask, request, abort
from flask_restx import Resource, Api, fields
from model import SearchModel
from waitress import serve


app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="MoJ Search",
    description="API for semantic document searching",
)

model = SearchModel()

chatbot_ns = api.namespace("search", description="Used to query the search")

question_model = api.model(
    "Question",
    {
        "QuestionText": fields.String,
    },
)

search_result = api.model(
    "SearchResult",
    {
        "AnswerTexts": fields.List(fields.List(fields.String)),
    },
)


@chatbot_ns.route("/query")
class Query(Resource):
    @api.response(200, "Success", search_result)
    @api.response(400, "Error")
    @api.expect(question_model)
    def post(self):
        try:
            data = request.get_json()
            text = data["QuestionText"]
            search_results = model.query(text)
            return {
                "SearchResult": {
                    "AnswerTexts": search_results,
                }
            }
        except Exception as e:
            abort(400, str(e))


@chatbot_ns.route("/train")
class Train(Resource):
    @api.response(200, "Success")
    @api.response(400, "Error")
    def post(self):
        try:
            model.train()
            return "Success", 200
        except Exception as e:
            abort(400, str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)
    # serve(app=app, host="0.0.0.0", port=7000)
