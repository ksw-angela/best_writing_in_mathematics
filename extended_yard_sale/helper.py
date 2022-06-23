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

        if any(value != 'int' for value in wealth.values()):
            raise Exception('start_wealth needs wealth represented as an integer')

        if len(wealth.keys()) % 2 != 0 or len(wealth.keys()) == 0:
            raise Exception('start_wealth needs an even number of people')

    def plot_wealth(self):
        pass


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
                 n=10000,
                 chi=0,
                 zeta=0,
                 kappa=0):
        """Initializes the extended yard sale model.

        Args:
            start_wealth (dict): Wealth of all transacting parties.
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
        super().__init__(start_wealth)
        self.win_percentage = win_percentage
        self.n = n
        self.chi = chi
        self.zeta = zeta
        self.kappa = kappa
        self._n_people = len(self.wealth.keys())

    def _update_average_wealth(self):
        """Calculates the average wealth of all people in the yard sale and updates the
        _avg_wealth attribute."""
        self._avg_wealth = np.mean(self.wealth.values())

    def _pay_tax(self):
        """Updates the wealth attribute in the following way:
        - People richer than the average wealth pay a flat wealth tax proportional
          to the chi attribute
        - People poorer than the average equally share the tax paid by the richer people
        """
        tax = self.chi * self.wealth.loc[self.wealth['wealth'] > self._avg_wealth, 'wealth'].sum()
        per_person_subsidy = round(
            tax / len(self.wealth.loc[self.wealth['wealth'] <= self._avg_wealth].index))
        self.wealth.loc[self.wealth['wealth'] > self._avg_wealth, 'wealth'] -= tax
        self.wealth.loc[self.wealth['wealth'] <= self._avg_wealth, 'wealth'] += per_person_subsidy

    def _loan_S(self):
        """Updates the wealth attribute for all people by providing a loan proportional to
        the attribute kappa before the transaction.
        """
        self.wealth.loc[:, 'wealth'] = self.wealth.loc[:, 'wealth'] + (
                    self.kappa * self._avg_wealth)

    def _collect_S(self):
        """Updates the wealth attribute for all people by paying back the loan after
        the transaction.
        """
        self.wealth.loc[:, 'wealth'] = self.wealth.loc[:, 'wealth'] - (
                    self.kappa * self._avg_wealth)

    def _add_pair_id(self):
        """Returns a dictionary of dictionaries with transacting members of form:
            {pair_id: {person_id: wealth}}
        The pair_id is shared among transacting pairs. For each pair_id, there should only be two
        person_id's.
        :return: object
        """
        wealth_permutation = self.wealth.sample(frac=1).reset_index(drop=True)
        wealth_permutation['group'] = list(range(int(self._n_people / 2))) * 2
        return wealth_permutation

    def _add_dice_roll(self, wealth_permutation):
        """Determine the winner of the transaction for all pairs.

        Include the bias that gives the wealthier person the advantage.
        Let 0 be a win for the richer person and 1 for the poorer person.

        :param wealth_permutation: dictionary of dictionaries of form:
            {pair_id: {person_id: wealth}}
        :return: object
        """
        wealth_difference = (wealth_permutation.groupby('group')['wealth'].max()
                             - wealth_permutation.groupby('group')['wealth'].min()).reset_index(
            drop=True)
        bias = 0.5 * (1 + self.zeta * wealth_difference)
        dice_rolls = []
        for i in bias:
            roll = np.random.choice(np.arange(0, 2), p=[i, 1 - i])
            dice_rolls.append(roll)
        wealth_permutation['dice_roll'] = dice_rolls * 2
        del wealth_difference, bias
        return wealth_permutation

    def _exchange_wealth(self, wealth_permutation):
        """Return the updated wealth with money moving from the loser to the winner.

        :param wealth_permutation: dictionary of dictionaries of form:
            {pair_id: {person_id: wealth}}
        :return: object
        """
        exchange_amount = self.win_percentage * wealth_permutation.groupby('group')['wealth'] \
            .min().reset_index(drop=True)
        poor = wealth_permutation.sort_values(['group', 'wealth']) \
            .drop_duplicates('group', keep='first').reset_index(drop=True)
        rich = wealth_permutation.sort_values(['group', 'wealth']) \
            .drop_duplicates('group', keep='last').reset_index(drop=True)
        poor['wealth'] += ((2 * poor['dice_roll'] - 1) * exchange_amount).round(2)
        rich['wealth'] += ((1 - 2 * rich['dice_roll']) * exchange_amount).round(2)
        wealth_permutation = pd.concat([poor, rich])
        del exchange_amount, poor, rich
        return wealth_permutation

    def perform_sale(self):
        """Runs a single iteration of the extended yard sale model and updates the wealth attribute.

        :return: None
        """
        self._update_average_wealth()
        self._pay_tax()
        self._loan_S()
        wealth_permutation = self._add_pair_id()
        wealth_permutation = self._add_dice_roll(wealth_permutation)
        wealth_permutation = self._exchange_wealth(wealth_permutation)
        self.wealth = wealth_permutation[['person', 'wealth']].reset_index(drop=True)
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
