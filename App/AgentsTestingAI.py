from abc import ABC, abstractmethod
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

if __name__ == '__main__':

    class TrainingEnvironment:
        def __init__(self):
            self.data = None
            self.data_type = None
            self.model = None
            self.metrics = {}
            self.goal_reached = False

            self.attempts = 0
            self.max_attempts = 5

    class TrainingBeliefs:
        def __init__(self):
            self.target_accuracy = 0.9


    class Agent:
        def __init__(self):
            self.alive = True
            self.beliefs = {}

        def step(self, env):
            raise NotImplementedError

    # Data Loaders

    class DataPackage:
        def __init__(self, raw, data_type):
            self.raw = raw
            self.data_type = data_type

    class DataLoader(ABC):
        @abstractmethod
        def can_handle(self, source) -> bool:
            pass

        @abstractmethod
        def load(self, source):
            pass


    class CSVLoader(DataLoader):
        def can_handle(self, source):
            return source.endswith(".csv")

        def load(self, source):
            print("[CSVLoader] Loading CSV data")
            # Data
            return {"type": "tabular", "data": [1, 2, 3, 4]}


    class JSONLoader(DataLoader):
        def can_handle(self, source):
            return source.endswith(".json")

        def load(self, source):
            print("[JSONLoader] Loading JSON data")
            return "numeric_data_from_json"


    class ImageLoader(DataLoader):
        def can_handle(self, source):
            return source.endswith((".png", ".jpg", ".jpeg"))

        def load(self, source):
            print("[ImageLoader] Loading image data")
            return "numeric_image_tensor"


    class DataAgent(Agent):

        def step(self, env: TrainingEnvironment):
            if env.data is not None:
                return  # already loaded

            source = self.beliefs["source"]

            if source.endswith(".csv"):
                env.data = DataPackage([1, 2, 3], "tabular")
            elif source.endswith(".png"):
                env.data = DataPackage([[0, 1], [1, 0]], "image")
            elif source.endswith(".wav"):
                env.data = DataPackage([0.1, 0.4], "sound")
            else:
                raise ValueError("Unsupported data")

            env.data_type = env.data.data_type
            print(f"[DataAgent] Loaded {env.data_type}")

    #############################################

    # Model types

    class ModelStrategy(ABC):
        supported_formats = set()

        @abstractmethod
        def train(self, data):
            pass

        @abstractmethod
        def evaluate(self, data):
            pass

        @abstractmethod
        def get_name(self):
            pass


    class BaseModel:
        supported_formats = set()

        def train(self, data):
            pass

        def test(self, data):
            pass

    import random

    class DenseModel(BaseModel):
        supported_formats = {"tabular", "sound"}

        def train(self, data):
            print("Training Dense model")

        def test(self, data):
            return random.uniform(0.7, 0.9)


    class ConvModel(BaseModel):
        supported_formats = {"image"}

        def train(self, data):
            print("Training Conv model")

        def test(self, data):
            return random.uniform(0.8, 0.95)


    class TrainingAgent(Agent):
        def __init__(self, models):
            super().__init__()
            self.models = models
            self.current_model_index = 0

        def step(self, env: TrainingEnvironment):
            if env.goal_reached:
                return

            if env.attempts >= env.max_attempts:
                print("[TrainingAgent] Max attempts reached ❌")
                self.alive = False
                return

            # Reset model if previous failed
            env.model = self.models[self.current_model_index % len(self.models)]
            self.current_model_index += 1

            env.attempts += 1

            print(f"[TrainingAgent] Attempt {env.attempts} using {env.model.__class__.__name__}")

            env.model.train(env.data.raw)
            acc = env.model.test(env.data.raw)

            env.metrics["accuracy"] = acc


    class EvaluationAgent(Agent):
        def step(self, env: TrainingEnvironment):
            if env.goal_reached:
                return

            acc = env.metrics.get("accuracy", 0)

            if acc >= self.beliefs["min_accuracy"]:
                env.goal_reached = True
                print(f"Accuracy: {acc}")
                print("[EvaluationAgent] Goal reached ✅")
            else:
                print(f"Accuracy: {acc}")
                print("[EvaluationAgent] Goal not reached, retrying...")

    class AccuracyGoal:
        def __init__(self, threshold):
            self.threshold = threshold

        def is_achieved(self, accuracy):
            return accuracy >= self.threshold


    class SaveAgent(Agent):
        def step(self, env: TrainingEnvironment):
            if env.goal_reached and env.model:
                print(f"[SaveAgent] Saving {env.model.__class__.__name__}")
                self.alive = False


    def run_system(agents, env, max_steps=20):
        for step in range(max_steps):
            print(f"\n--- Step {step} ---")

            for agent in agents:
                if agent.alive:
                    agent.step(env)
                else:
                    print("Steps tresholds is reached, goal not reached, stopping learning process...")
                    return

            if env.goal_reached:
                print("Stopping agent cycle ✔")
                break


    env = TrainingEnvironment()

    data_agent = DataAgent()
    data_agent.beliefs["source"] = "image.png"

    train_agent = TrainingAgent([
        DenseModel(),
        ConvModel()
    ])

    eval_agent = EvaluationAgent()
    eval_agent.beliefs["min_accuracy"] = 0.95

    save_agent = SaveAgent()

    agents = [
        data_agent,
        train_agent,
        eval_agent,
        save_agent
    ]

    run_system(agents, env)