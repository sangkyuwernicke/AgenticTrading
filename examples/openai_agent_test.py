from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv()

agent = Agent(name="Assistant", instructions="당신은 도움이 되는 어시스턴트입니다.")

result = Runner.run_sync(agent, "프로그래밍에서 재귀에 대한 한국 전통 시조를 지어줘.")
print(result.final_output)

