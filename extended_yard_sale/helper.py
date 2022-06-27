import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


class Wealth:
    """Stores the wealth of an even number of people as a dictionary.

        Args:
            start_wealth (dict): Wealth for an even number of people

        Attributes:
            start_wealth (dict): Wealth for an even number of people

    """
    def __init__(self, start_wealth):
        self._validate_wealth(start_wealth)
        self.wealth = start_wealth

    @staticmethod
    def _validate_wealth(wealth):
        """Validates the wealth dictionary.

        :param wealth: dictionary
        :return: None
        """
        if not isinstance(wealth, dict):
            raise Exception('start_wealth needs to be a dictionary')

        if len(wealth.keys()) % 2 != 0 or len(wealth.keys()) == 0:
            raise Exception('start_wealth needs an even number of people')

    def plot_wealth(self):
        """Plots the wealth dictionary as a histogram.

        :return: None
        """
        plt.bar(self.wealth.keys(), self.wealth.values())
        plt.show()


class ExtendedYardSale(Wealth):
    """Conducts the extended yard sale for a given wealth dictionary.

        Attributes:
            wealth (dict): Wealth of all transacting parties.
            win_percentage (float): Percentage of the poorer person's wealth that is moved from the
                loser of the transaction to the winner.
            n (int): Number of transactions. Defaults to 10000.
            chi (float): Tax rate paid before each transaction by people who's wealth exceeds the
                average wealth of the system and is distributed equally to people who's wealth
                is below the average wealth of the system. Defaults to 0, or no tax being paid.
            zeta (float): Proportional to the amount we bias the coin flip in favour of the richer
                person. The bias is equal to:
                    zeta * (absolute difference in wealth of the transacting parties)
                Defaults to 0, or no bias for the richer person.
            kappa (float): Proportional to the amount of negative wealth a person can have.
                The wealth of the poorest person at any time is -S, where:
                    S = kappa * average wealth
                Before each transaction, we loan S to both agents so both have positive wealth.
                After the transaction is complete, both parties repay their debt of S.

    """
    def __init__(self,
                 start_wealth,
                 win_percentage,
                 chi=0,
                 zeta=0,
                 kappa=0,
                 n=10000,
                 seed=0):
        """Initializes the extended yard sale model.

        Args:
            start_wealth (dict): Wealth of all transacting parties.
            win_percentage (float): Percentage of the poorer person's wealth that is moved from the
                loser of the transaction to the winner.
            chi (float): Tax rate paid before each transaction by people who's wealth exceeds the
                average wealth of the system and is distributed equally to people who's wealth
                is below the average wealth of the system. Defaults to 0, or no tax being paid.
            zeta (float): Proportional to the amount we bias the coin flip in favour of the richer
                person. The bias is equal to:
                    zeta * (absolute difference in wealth of the transacting parties)
                Defaults to 0, or no bias for the richer person.
            kappa (float): Proportional to the amount of negative wealth a person can have.
                The wealth of the poorest person at any time is -S, where:
                    S = kappa * average wealth
                Before each transaction, we loan S to both agents so both have positive wealth.
                After the transaction is complete, both parties repay their debt of S.
            n (int): Number of transactions. Defaults to 10000.
            seed (int): Optional random seed.

        """
        super().__init__(start_wealth)
        self.win_percentage = win_percentage
        self.chi = chi
        self.zeta = zeta
        self.kappa = kappa
        self.n = n
        self._n_people = len(self.wealth.keys())
        random.seed(seed)

    def _update_average_wealth(self):
        """Calculates the average wealth of all people in the yard sale and updates the
        _avg_wealth attribute."""
        self._avg_wealth = np.mean(list(self.wealth.values()))

    def _pay_tax(self):
        """Updates the wealth attribute in the following way:
        - People richer than the average wealth pay a flat wealth tax proportional
          to the chi attribute
        - People poorer than the average equally share the tax paid by the richer people
        """
        rich = {p: w for p, w in self.wealth.items() if w > self._avg_wealth}
        poor = {p: w for p, w in self.wealth.items() if w <= self._avg_wealth}

        tax = self.chi * np.sum(list(rich.values()))

        if len(poor.keys()) > 0:
            per_person_subsidy = round(tax / len(poor.keys()))
            poor = {p: w + per_person_subsidy for p, w in poor.items()}

        if len(rich.keys()) > 0:
            rich = {p: w - (w * tax) for p, w in rich.items()}

        self.wealth = {**rich, **poor}

    def _loan_S(self):
        """Updates the wealth attribute for all people by providing a loan proportional to
        the attribute kappa before the transaction.
        """
        self.wealth = {p: w + (self.kappa * self._avg_wealth) for p, w in self.wealth.items()}

    def _collect_S(self):
        """Updates the wealth attribute for all people by paying back the loan after
        the transaction.
        """
        self.wealth = {p: w - (self.kappa * self._avg_wealth) for p, w in self.wealth.items()}

    def _pair_people(self):
        """Returns a list of dictionaries with transacting members.
        Each element of the list represents a pair of people that will transact.

        :return: object
        """
        people = list(self.wealth.keys())
        pair_ids = np.array(random.sample(people, self._n_people)).reshape(-1, 2)
        pair_wealths = []
        for p in pair_ids:
            pair_wealth = {k: self.wealth[k] for k in p}
            pair_wealths.append(pair_wealth)
        return pair_wealths

    def _roll_dice(self, pair_wealths):
        """Determine the winner of the transaction for all pairs.

        Include the bias that gives the wealthier person the advantage.
        Let 0 be a win for the richer person and 1 for the poorer person.

        :param pair_wealths: list of dictionaries
        :return: object
        """
        dice_rolls = []
        for p in pair_wealths:
            wealth_values = list(p.values())
            wealth_difference = abs(wealth_values[0] - wealth_values[1])
            bias = self.zeta * wealth_difference
            dice_rolls.append(random.choices(range(1),
                                             [(0.5 + bias) / (1 + bias), 0.5 / (1 + bias)]))
        return dice_rolls

    def _exchange_wealth(self, wealth_permutation, dice_rolls):
        """Return the updated wealth with money moving from the loser to the winner.

        :param wealth_permutation: dictionary of dictionaries of form:
            {pair_id: {person_id: wealth}}
        :param dice_rolls: dictionary of dice rolls
        :return: object
        """
        wealth = {}
        for pair, roll in zip(wealth_permutation, dice_rolls):
            exchange_amount = np.mean(list(pair.values())) * self.win_percentage
            poor = min(pair, key=pair.get)
            rich = max(pair, key=pair.get)
            wealth[poor] = pair[poor] + ((2 * roll - 1) * exchange_amount).round(2)
            wealth[rich] = pair[rich] + ((1 - 2 * roll) * exchange_amount).round(2)
        return wealth

    def perform_sale(self):
        """Runs a single iteration of the extended yard sale model and updates the wealth attribute.

        :return: None
        """
        self._update_average_wealth()
        self._pay_tax()
        self._loan_S()
        wealth_permutation = self._pair_people()
        dice_rolls = self._roll_dice(wealth_permutation)
        wealth = self._exchange_wealth(wealth_permutation, dice_rolls)
        self.wealth = wealth
        self._collect_S()

    def run_yard_sale(self, plot_n=1000, plot=True):
        """Runs multiple iterations of the extended yard sale model.

        :param plot_n: (int) The number of iterations that need to elapse before plotting the wealth
            distribution. Default to 1000.
        :param plot: (booelan) If True, plots are returned. Default to True.
        :return: None
        """
        if plot & (plot_n > self.n):
            raise Exception('plot_n needs to be lower than n')
        for i in tqdm(range(self.n)):
            self.perform_sale()
            if plot & (i % plot_n == 0):
                super().plot_wealth()

    def _get_sale_stats(self):
        pass

    def run_multiple_sales(self):
        pass
