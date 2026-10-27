"""LangGraph CLI entry point; use LangSmith Studio to interact with the graph."""

from parking_agent import ParkingSpaceReservationAgent

parking_agent = ParkingSpaceReservationAgent()
agent = parking_agent.graph
