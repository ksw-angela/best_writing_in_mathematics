import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


class Wealth:
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

        if len(wealth.keys()) % 2 != 0:
            raise Exception('start_wealth needs an even number of people')

    def plot_wealth(self):
        pass


class ExtendedYardSale(Wealth):
    def __init__(self,
                 start_wealth,
                 win_percentage: float,
                 n=10000,
                 chi=0,
                 zeta=0,
                 kappa=0):
        super().__init__(start_wealth)
        self.win_percentage = win_percentage
        self.n = n
        self.chi = chi
        self.zeta = zeta
        self.kappa = kappa
        self._n_people = len(self.wealth.index)

    def _update_average_wealth(self):
        # calculate average wealth of all people in the yard sale
        self._avg_wealth = self.wealth['wealth'].mean()

    def _pay_tax(self):
        # people that are wealthier than the average pay the flat wealth tax
        # tax is distributed equally among people that are poorer than the average
        tax = self.chi * self.wealth.loc[self.wealth['wealth'] > self._avg_wealth, 'wealth'].sum()
        per_person_subsidy = round(
            tax / len(self.wealth.loc[self.wealth['wealth'] <= self._avg_wealth].index))
        self.wealth.loc[self.wealth['wealth'] > self._avg_wealth, 'wealth'] -= tax
        self.wealth.loc[self.wealth['wealth'] <= self._avg_wealth, 'wealth'] += per_person_subsidy

    def _loan_S(self):
        # Loan S to everyone before the transaction
        self.wealth.loc[:, 'wealth'] = self.wealth.loc[:, 'wealth'] + (
                    self.kappa * self._avg_wealth)

    def _collect_S(self):
        # Both parties pay back loan of S at the end of transaction
        self.wealth.loc[:, 'wealth'] = self.wealth.loc[:, 'wealth'] - (
                    self.kappa * self._avg_wealth)

    def _add_pair_id(self):
        # match people by randomly shuffling people add a column 'group' to represent the pair id
        wealth_permutation = self.wealth.sample(frac=1).reset_index(drop=True)
        wealth_permutation['group'] = list(range(int(self._n_people / 2))) * 2
        return wealth_permutation

    def _add_dice_roll(self, wealth_permutation):
        # Roll dice for all pairs using the bias that gives the wealthier person the advantage
        # Let 0 be a win for the richer person and 1 be a win for the poorer person
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
        # Move money from the loser to the winner
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
        self._update_average_wealth()
        self._pay_tax()
        self._loan_S()
        wealth_permutation = self._add_pair_id()
        wealth_permutation = self._add_dice_roll(wealth_permutation)
        wealth_permutation = self._exchange_wealth(wealth_permutation)
        self.wealth = wealth_permutation[['person', 'wealth']].reset_index(drop=True)
        self._collect_S()

    def run_yard_sale(self, plot_n=1000, plot=True):
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
