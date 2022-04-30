from typing import Optional, Tuple

import plotly.graph_objects as go
import numpy as np


def map_range(x: float, input: Tuple[float, float], output: Tuple[float, float]) -> float:
    return (x - input[0]) * (output[1] - output[0]) / (input[1] - input[0]) + output[0]


class Pendulum:
    """
    A Simple pendulum class.

    """

    def __init__(self, origin: Tuple[float, float] = (0.0, 0.0),
                 mass: float = 1.0,
                 length: float = 1.0,
                 initial_conditions: float = np.random.rand(1),
                 gravity: float = 9.81) -> None:
        self.origin = origin
        self.mass = mass
        self.length = length
        self.initial_conditions = map_range(
            initial_conditions, (0, 1), (-np.pi, np.pi))
        self.gravity = gravity
        self.time = 0.0
        self.delta_t = 0.1

        # State variables
        self.theta = self.initial_conditions
        self.omega = 0.0
        self.alpha = 0.0

        # Derived variables
        self.x: float = self.origin[0] + self.length * np.sin(self.theta)

        self.y: float = self.origin[1] + self.length * np.cos(self.theta)

    def step(self) -> Tuple[float, float]:
        """
        Perform a single step of the simulation.
        """
        self.time += self.delta_t
        self.alpha = -self.gravity / self.length * np.sin(self.theta)
        self.omega += self.alpha * self.delta_t
        self.theta += self.omega * self.delta_t

        # Update state variables
        self.update_xy(self.theta)

        return self.x, self.y

    def update_xy(self, theta: float) -> Tuple[float, float]:
        """
        Update the x and y coordinates of the pendulum.
        """
        self.x = self.origin[0] + self.length * np.sin(theta)
        self.y = self.origin[1] + self.length * np.cos(theta)
        return self.x, self.y

    def simulation(self, render: bool = False, max_steps: int = 100, dt: Optional[float] = None) -> Tuple[float, float]:
        """
        Run the simulation.
        """
        if dt is not None:
            self.delta_t = dt
        for i in range(max_steps):
            self.step()
            print("X:", self.x, "Y:", self.y)
            if render:
                self.render()

    def render(self) -> None:
        """
        Render the simulation.
        """
        fig = go.Figure(
            data=[go.Scatter(x=[0, 1], y=[0, 1])],
            layout=go.Layout(
                xaxis=dict(range=[0, 5], autorange=False),
                yaxis=dict(range=[0, 5], autorange=False),
                title="Start Title",
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None])])]
            ),
            frames=[go.Frame(data=[go.Scatter(x=[1, 2], y=[1, 2])]),
                    go.Frame(data=[go.Scatter(x=[1, 4], y=[1, 4])]),
                    go.Frame(data=[go.Scatter(x=[3, 4], y=[3, 4])],
                             layout=go.Layout(title_text="End Title"))]
        )

        fig.show()


def main():
    pend = Pendulum()

    pend.simulation()


if __name__ == "__main__":
    main()
    print("Done.")
    input("Press enter to exit.")
    exit(0)
