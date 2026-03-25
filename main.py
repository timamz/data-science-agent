import logging

from agents.orchestrator import OrchestratorAgent


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    agent = OrchestratorAgent()
    agent.run(time_budget=600)


if __name__ == "__main__":
    main()
