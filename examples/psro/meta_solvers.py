import abc
import logging
import signal
from contextlib import contextmanager

import nashpy as nash
import numpy as np


logger = logging.getLogger(__name__)


@contextmanager
def timeout_context(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f'block timedout after {duration} seconds')

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


class Solver(metaclass=abc.ABCMeta):
    NAME: str
    ABBREVIATED_NAME: str

    def __init__(self, payoff_matrices):
        self.payoff_matrices = np.asarray(payoff_matrices)
        assert payoff_matrices.shape[0] == 2

        self.game = nash.Game(*payoff_matrices)

    @abc.abstractmethod
    def solve(self):
        raise NotImplementedError

    def __call__(self):
        return self.solve()


class NashEquilibrium(Solver):
    NAME = 'NashEquilibrium'
    ABBREVIATED_NAME = 'NE'
    TIMEOUT = 300  # seconds
    ITERATIONS = 100000  # fallback iterations for fictitious play

    def solve(self):
        logger.info(str(self.game))

        for name, method in (
            ('support enumeration', self.game.support_enumeration),
            ('vertex enumeration', self.game.vertex_enumeration),
            ('Lemke Howson algorithm', self.game.lemke_howson_enumeration),
        ):
            try:
                logger.info('Trying to solve the game with with %s.', name)
                with timeout_context(self.TIMEOUT):
                    sigma_row, sigma_col = next(method())
            except (StopIteration, RuntimeError):
                logger.info('Failed to solve the game with with %s.', name)
            except TimeoutError:
                logger.info('Maximum execution time exceeded with %s.', name)
            else:
                logger.info(
                    'Got Nash equilibria:\nrow player: %s\ncolumn player: %s', sigma_row, sigma_col
                )
                return sigma_row, sigma_col

        *_, (sigma_row, sigma_col) = iter(self.game.fictitious_play(iterations=self.ITERATIONS))
        sigma_row = np.asarray(sigma_row) / np.sum(sigma_row)
        sigma_col = np.asarray(sigma_col) / np.sum(sigma_col)

        logger.info(
            'Got approximate Nash equilibria using fictitious play:\nrow player: %s\ncolumn player: %s',
            sigma_row,
            sigma_col,
        )
        return sigma_row, sigma_col


class SelfPlay(Solver):
    NAME = 'SelfPlay'
    ABBREVIATED_NAME = 'SP'

    def solve(self):
        sigma_row = np.zeros((self.payoff_matrices.shape[1],), dtype=np.float64)
        sigma_col = np.zeros((self.payoff_matrices.shape[2],), dtype=np.float64)

        sigma_row[-1] = 1.0
        sigma_col[-1] = 1.0

        return sigma_row, sigma_col


class FictitiousSelfPlay(Solver):
    NAME = 'FictitiousSelfPlay'
    ABBREVIATED_NAME = 'FSP'

    def solve(self):
        sigma_row = np.ones((self.payoff_matrices.shape[1],), dtype=np.float64)
        sigma_col = np.ones((self.payoff_matrices.shape[2],), dtype=np.float64)

        sigma_row /= sigma_row.sum()
        sigma_col /= sigma_col.sum()

        return sigma_row, sigma_col


META_SOLVERS = {}
for solver in (NashEquilibrium, SelfPlay, FictitiousSelfPlay):
    META_SOLVERS[solver.NAME] = solver
    META_SOLVERS[solver.ABBREVIATED_NAME] = solver

del solver
