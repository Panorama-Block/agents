from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from agents.tools.custom_tool import GeminiImageDirectTool, GrokSearchTool

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Agents():
	"""Agents crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def researcher(self) -> Agent:
			return Agent(
			config=self.agents_config['researcher'],
			verbose=True
		)

	@agent
	def avax_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['avax_researcher'],
			tools=[GrokSearchTool()],
			verbose=True
		)

	@agent
	def hedera_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['hedera_researcher'],
			tools=[GrokSearchTool()],
			verbose=True
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True
		)

	@agent
	def twitter_redactor(self) -> Agent:
			return Agent(
			config=self.agents_config['twitter_redactor'],
			verbose=True
		)

	@agent
	def image_generator(self) -> Agent:
		return Agent(
			config=self.agents_config['image_generator'],
			tools=[GeminiImageDirectTool()],
			verbose=True
		)
   

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def avax_research_task(self) -> Task:
		return Task(
			config=self.tasks_config['avax_research_task']
		)

	@task
	def hedera_research_task(self) -> Task:
		return Task(
			config=self.tasks_config['hedera_research_task']
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='zico_report.md'
		)
  
	@task
	def avax_reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='avax_report.md'
		)

	@task
	def hedera_reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='hedera_report.md'
		)
  
	@task
	def twitter_redaction_task(self) -> Task:
		return Task(
			config=self.tasks_config['twitter_redaction_task'],
			output_file='tweet.md'
		)

	@task
	def image_generation_task(self) -> Task:
		return Task(
			config=self.tasks_config['image_generation_task']
		)

	@crew
	def tweet_crew(self) -> Crew:
		"""Creates the Tweet Generation crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=[self.researcher(), self.reporting_analyst(), self.twitter_redactor()],
			tasks=[self.research_task(), self.reporting_task(), self.twitter_redaction_task()],
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
  
	@crew
	def image_crew(self) -> Crew:
		"""Creates the Image Generation crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=[self.image_generator()],
			tasks=[self.image_generation_task()],
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

	@crew
	def avax_crew(self) -> Crew:
		"""Creates the Avax Research and Analysis crew"""
		return Crew(
			agents=[
				self.avax_researcher(), 
				self.reporting_analyst(), 
				self.twitter_redactor()
			],
			tasks=[
				self.avax_research_task(), 
				self.avax_reporting_task(), 
				self.twitter_redaction_task(),
			],
			process=Process.sequential
		)

	@crew
	def hedera_crew(self) -> Crew:
		"""Creates the Hedera Research and Analysis crew"""
		return Crew(
			agents=[
				self.hedera_researcher(), 
				self.reporting_analyst(), 
				self.twitter_redactor()
			],
			tasks=[
				self.hedera_research_task(), 
				self.hedera_reporting_task(), 
				self.twitter_redaction_task(),
			],
			process=Process.sequential
		)