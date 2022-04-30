from typing import List, Optional, Tuple

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
        self.origin: Tuple[float, float] = origin
        self.mass: float = mass
        self.length: float = length
        self.initial_conditions: float = map_range(
            initial_conditions, (0, 1), (-np.pi, np.pi))
        self.gravity: float = gravity
        self.time: float = 0.0
        self.delta_t: float = 0.1
        self.history: List[Tuple[float, float]] = []

        # State variables
        self.theta: float = self.initial_conditions
        self.omega: float = 0.0
        self.alpha: float = 0.0

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
            self.history.append((self.x, -self.y))

        if render:
            self.render(max_steps=max_steps)

    def render(self, max_steps: int) -> None:
        """
        Render the simulation.
        """
        fig = go.Figure(
            data=[go.Scatter(x=[0, 0], y=[0, -1])],
            layout=go.Layout(
                xaxis=dict(range=[-3, 3], autorange=False),
                yaxis=dict(range=[-3, 3], autorange=False),
                title="Start Title",
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None, {"frame": {"duration": 50,
                                                         "redraw": False},
                                               "fromcurrent": True,
                                               "transition": {"duration": 0}}])])]
            ),
            frames=[go.Frame(data=[go.Scatter(x=[self.origin[0], *self.history[i][0]], y=[
                             self.origin[1], *self.history[i][1]])]) for i in range(max_steps)]
        )

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

        fig.show()


def main():
    pend = Pendulum()

    pend.simulation(True, max_steps=1000, dt=0.01)
    frames = [go.Frame(data=[go.Scatter(x=[pend.origin[0], *pend.history[i][0]], y=[
        pend.origin[1], *pend.history[i][1]])]) for i in range(100)]
    print(frames)


if __name__ == "__main__":
    main()
    print("Done.")
    input("Press enter to exit.")
    exit(0)
