from xml.dom.minidom import parse
import xml.dom.minidom
import itertools
from random import randrange
import math
import negoAction


class UtilitySpace:
    def __init__(self, profile_file):
        # Role of the agent.
        self.profile_file = profile_file
        # Role weights of the agent. [0.5, 0.7, 0.2] etc.
        self.issue_weights = []
        # Name to index dict for issue. {"Apple": 0, "Banana": 1} etc.
        self.issue_name_to_index = {}
        # Index to name dict for issue. {0: "Apple", 1: "Banana"} etc.
        self.issue_index_to_name = {}
        # Issue's value evaluations.  { "Apple": {0: 0.3, 1: 0.2}, "Banana": {0: 0.2, 1: 0.7} } etc.
        self.issue_value_evaluation = {}
        # Keep every issue's values in a list. [ [0, 1, 2, 3], ["Antalya", "Izmir"] ] etc.
        self.issue_values_list = []
        # Issue name list.
        self.issue_names = []
        # Issues max count list for each issue.
        self.issue_max_counts = []
        # Call the role weight function for complete the weight dictionary.
        self.__set_utility_space()
        # Generate all possible offers.
        self.__generate_all_possible_offers()

    # Read the xml file of weights of keywords, then add to dictionary for calculation purpose.
    def __set_utility_space(self):
        # Open XML document using minidom parser
        DOMTree = xml.dom.minidom.parse(self.profile_file)
        collection = DOMTree.documentElement
        # Get issue list from the preference profile.
        issue_list = collection.getElementsByTagName("issue")
        weight_list = collection.getElementsByTagName("weight")
        # Get weight of the keywords and append to role weights.
        for issue, weight in zip(issue_list, weight_list):
            issue_name = issue.attributes["name"].value
            issue_index = issue.attributes["index"].value
            issue_max_count = issue.attributes["max_count"].value
            issue_weight = weight.attributes["value"].value
            self.issue_name_to_index[issue_name] = int(issue_index) - 1
            self.issue_index_to_name[int(issue_index) - 1] = issue_name
            self.issue_weights.append(float(issue_weight))
            self.issue_names.append(issue_name)
            self.issue_max_counts.append(int(issue_max_count))
            value_eval_dict = {}
            issue_values = []
            for item in issue.getElementsByTagName("item"):
                item_value = item.getAttribute("value")
                item_eval = item.getAttribute("evaluation")
                value_eval_dict[item_value] = float(item_eval)
                issue_values.append(item_value)
            self.issue_value_evaluation[issue_name] = value_eval_dict
            self.issue_values_list.append(issue_values)
        # Keep total count of the issues.
        self.issues_total_count = sum(self.issue_max_counts)

    def __generate_all_possible_offers(self):
        """
        Generate all possible offers and utilities in the ascending order.
        """
        self.all_possible_offers = list(itertools.product(*self.issue_values_list))
        self.all_possible_offers = [
            [int(float(j)) for j in i] for i in self.all_possible_offers
        ]
        self.__all_possible_offers_utilities = [
            self.get_offer_utility(offer) for offer in self.all_possible_offers
        ]

    # Calculate the offer's point -> either as list [0, 3, 5] or negoAction.Offer object.
    def get_offer_utility(self, offer):
        offer_utility = 0
        if isinstance(offer, negoAction.Offer):
            for issue_index, item in enumerate(offer.get_bid()):
                issue_name = self.issue_index_to_name[issue_index]
                issue_weight = self.issue_weights[issue_index]
                item_eval = self.issue_value_evaluation[issue_name][str(item)]
                offer_utility += issue_weight * item_eval
        else:
            for issue_index, item in enumerate(offer):
                issue_name = self.issue_index_to_name[issue_index]
                issue_weight = self.issue_weights[issue_index]
                item_eval = self.issue_value_evaluation[issue_name][str(item)]
                offer_utility += issue_weight * item_eval
        return round(offer_utility, 6)

    def get_offer_between_utility(
        self, lower_utility_threshold, upper_utility_threshold
    ):
        filtered_offers = []
        while len(filtered_offers) == 0:
            filtered_offers = [
                x
                for x in self.all_possible_offers
                if (
                    self.get_offer_utility(x) >= lower_utility_threshold
                    and self.get_offer_utility(x) < upper_utility_threshold
                )
            ]
            upper_utility_threshold += 0.01
            lower_utility_threshold -= 0.01
        random_offer_index = randrange(0, len(filtered_offers))
        random_offer = filtered_offers[random_offer_index]
        return random_offer

    def get_offer_below_utility(self, target_utility):
        # Try to get lower_utility and upper boundry of the target utility that exist, if any exception happends set limits to 1 and max.
        try:
            lower_utility = max(
                [
                    utility
                    for index, utility in enumerate(
                        self.__all_possible_offers_utilities
                    )
                    if target_utility >= utility
                ]
            )
            lower_utility = max(lower_utility, 0.4)
            lower_utility = min(lower_utility, 0.9)
        except:
            lower_utility = 0.4

        try:
            upper_utility = min(
                [
                    utility
                    for index, utility in enumerate(
                        self.__all_possible_offers_utilities
                    )
                    if target_utility <= utility
                ]
            )
            upper_utility = min(upper_utility, 0.9)
        except:

            upper_utility = 0.9

        # If the target utility is the same as upper limit return it, otherwise return lower utility threshold.
        if target_utility == upper_utility:
            selected_utility = upper_utility
        else:
            selected_utility = lower_utility

        # Return the offer with the most element.
        indexes = [
            index
            for index, utility in enumerate(self.__all_possible_offers_utilities)
            if utility == selected_utility
        ]
        min_count = self.issues_total_count
        min_count_offer_index = -1
        for index in indexes:
            offer = self.all_possible_offers[index]
            offer = [int(issue) for issue in offer]
            if sum(offer) < min_count:
                min_count = sum(offer)
                min_count_offer_index = index

        return list(self.all_possible_offers[min_count_offer_index])

    def get_offer_above_utility(self, lower_utility_threshold):
        upper_utility_threshold = 1
        filtered_offers = []
        while len(filtered_offers) == 0:
            filtered_offers = [
                x
                for x in self.all_possible_offers
                if self.get_offer_utility(x) >= lower_utility_threshold
                and self.get_offer_utility(x) <= upper_utility_threshold
            ]
            upper_utility_threshold += 0.01
            lower_utility_threshold += 0.01
        random_offer_index = randrange(0, len(filtered_offers))
        random_offer = filtered_offers[random_offer_index]
        return random_offer

    def calculate_offer_for_opponent(self, offer):
        """
        Gets the offer and calculates leftover items for opponent in resource allocation type domains.
        """
        opponent_offer = []
        for issue_index, offered_amount in enumerate(offer):
            # Get issue name.
            issue_name = self.issue_index_to_name[issue_index]
            # Get max count of the issue.
            issue_max_count = self.issue_max_counts[issue_index]
            # Calculate leftover for the opponent of iterating issue.
            left_count = issue_max_count - int(offered_amount)
            opponent_offer.append(int(left_count))
        return opponent_offer

    def get_domain_type(self):
        return self.domain_type
