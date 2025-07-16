from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool

@CrewBase
class SocialMediaCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def topic_researcher(self):
        return Agent(config=self.agents_config["topic_researcher"], tools=[SerperDevTool()])

    @agent
    def post_writer(self):
        return Agent(config=self.agents_config["post_writer"])

    @agent
    def critic(self):
        return Agent(config=self.agents_config["critic"])

    @task
    def research_task(self):
        return Task(config=self.tasks_config["research_task"], agent=self.topic_researcher())

    @task
    def write_task(self):
        return Task(config=self.tasks_config["write_task"], agent=self.post_writer())

    @task
    def critique_task(self):
        return Task(config=self.tasks_config["critique_task"], agent=self.critic())

    @crew
    def crew(self):
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential)
