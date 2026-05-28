from abc import ABC, abstractmethod

if __name__ == '__main__':
    class Agent(ABC):
        def __init__(self, name):
            self.name = name
            self.alive = True

        @abstractmethod
        def perceive(self, env):
            pass

        @abstractmethod
        def decide(self):
            pass

        @abstractmethod
        def act(self, env):
            pass

        @abstractmethod
        def check_goals(self):
            pass

        def step(self, env):
            self.perceive(env)
            self.decide()
            self.act(env)
            self.check_goals()


    class BeliefBase:
        def __init__(self):
            self.facts = {}

        def update(self, key, value):
            self.facts[key] = value

        def get(self, key, default=None):
            return self.facts.get(key, default)


    """
    GOALS
    """


    class Goal:
        def __init__(self, description, terminal=False):
            self.description = description
            self.terminal = terminal

        def is_achieved(self, beliefs):
            raise NotImplementedError


    class StayAliveGoal(Goal):
        def is_achieved(self, beliefs):
            return beliefs.get("health") > 50


    class ReachDestinationGoal(Goal):
        def is_achieved(self, beliefs):
            return beliefs.get("destinationDistance") >= 100


    """
    INTENTIONS
    """
    class Intention:
        def execute(self, agent, env):
            pass

    class HealIntention(Intention):
        def execute(self, agent, env):
            health = agent.beliefs.get("health")

            if health > 80:  # healing goal achieved
                print(f"{agent.name} finished healing")
                agent.current_intention = None
                return

            print(f"{agent.name} heals")
            agent.beliefs.update("health", health + 30)
            print(f"Distance: {agent.beliefs.get('destinationDistance')}, Health: {agent.beliefs.get('health')}")

    class AdvanceIntention(Intention):
        def execute(self, agent, env):
            if agent.beliefs.get("health") < 50:
                agent.current_intention = None
                return

            if agent.beliefs.get("destinationDistance") < 100:
                print(f"{agent.name} goes forward, health decreases!")
                agent.beliefs.update("destinationDistance", agent.beliefs.get("destinationDistance") + 20 )
                agent.beliefs.update("health", agent.beliefs.get("health") - 15)
                print(f"Distance: {agent.beliefs.get('destinationDistance')}, Health: {agent.beliefs.get('health')}")
            else:
                print(f"{agent.name} destination reached!")
                agent.current_intention = None
                # agent.alive = False
                # return


    """
    DECISIONS
    """
    class SimpleDecisionStrategy:
        def choose_intention(self, agent):
            health = agent.beliefs.get("health")
            distance = agent.beliefs.get("destinationDistance")
            if health < 50:
                return HealIntention()
            elif distance <= 100:
                return AdvanceIntention()
            return None


    class SimpleAgent(Agent):
        def __init__(self, name):
            super().__init__(name)
            self.beliefs = BeliefBase()
            self.beliefs.update("health", 80)
            self.beliefs.update("destinationDistance", 0)
            self.goals = [StayAliveGoal("Stay alive", terminal=False)]
            self.goals = [ReachDestinationGoal("Reach destination", terminal=True)]
            self.decision_strategy = SimpleDecisionStrategy()
            self.current_intention = None

        def perceive(self, env):
            pass  # environment sensing could go here

        def decide(self):
            if self.beliefs.get("health") < 50:
                if not isinstance(self.current_intention, HealIntention):
                    self.current_intention = HealIntention()
                return

                # Otherwise, deliberate only if free
            if self.current_intention is None:
                self.current_intention = self.decision_strategy.choose_intention(self)

        def act(self, env):
            if self.current_intention:
                self.current_intention.execute(self, env)

        def check_goals(self):
            for goal in self.goals:
                if goal.terminal and goal.is_achieved(self.beliefs):
                    print(f"{self.name} achieved goal: {goal.description}")
                    self.alive = False
                    return


    class Message:
        def __init__(self, sender, receiver, performative, content):
            self.sender = sender
            self.receiver = receiver
            self.performative = performative
            self.content = content


    class CommunicatingAgent(SimpleAgent):
        def __init__(self, name):
            super().__init__(name)
            self.mailbox = []

        def receive(self, message):
            self.mailbox.append(message)

        def perceive(self, env):
            while self.mailbox:
                msg = self.mailbox.pop(0)
                print(f"{self.name} received message from {msg.sender}: {msg.content}")


    class Environment:
        def __init__(self):
            self.agents = []

        def add_agent(self, agent):
            self.agents.append(agent)

        def step(self):
            for agent in self.agents:
                if agent.alive:
                    agent.step(self)

    env = Environment()

    agent_a = CommunicatingAgent("Agent A")
    agent_b = CommunicatingAgent("Agent B")

    env.add_agent(agent_a)
    env.add_agent(agent_b)

    agent_a.receive(Message("Agent B", "Agent A", "inform", f"Hello! "
                            f"Starting distance: {agent_b.beliefs.get('destinationDistance')}, "
                            f"Starting health: {agent_b.beliefs.get('health')}"))
    agent_b.receive(Message("Agent A", "Agent B", "inform", f"Hi! "
                            f"Starting distance: {agent_b.beliefs.get('destinationDistance')}, "
                            f"Starting health: {agent_b.beliefs.get('health')}"))

    for i in range(10):
        if agent_a.alive and agent_b.alive:
            print(f"\n--- Step {i} ---")
            env.step()


