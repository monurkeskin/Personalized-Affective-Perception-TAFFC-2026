from HATN.utilitySpace import UtilitySpace
from ComparisonObject import ComparisonObject
from math import sqrt
import numpy as np
import copy


class UncertaintyModule:
    def __init__(self, utility_space):
        self.utility_space = utility_space

    def estimate_opponent_preferences(self, opponentOfferHistory):
        """
        This method takes opponent's bidding history and returns estimated utility_space of the opponent.
        """

        # Dictionary that keeps number of same values as keys, and compared offers list as values.
        comparablesDict = {}

        for i in range(len(opponentOfferHistory)):
            for j in range(i + 1, len(opponentOfferHistory)):
                # Compare current and next bid's issue values. If they are same, add to the list.
                same_values_indices = [
                    index
                    for index, (a, b) in enumerate(
                        zip(
                            opponentOfferHistory[i].get_bid(),
                            opponentOfferHistory[j].get_bid(),
                        )
                    )
                    if a == b
                ]
                # Keep number of values that are same.
                sameValueCount = len(same_values_indices)
                # If count does not exist, create new empty list to append later on.
                if sameValueCount not in comparablesDict:
                    comparablesDict[sameValueCount] = []
                # Create comparison variable for history offers.
                comparision_pair = ComparisonObject(
                    opponentOfferHistory[i], opponentOfferHistory[j]
                )
                # Check if this pair already exists in dict, append otherwise with same value count as a key.
                if comparision_pair not in comparablesDict[sameValueCount]:
                    comparablesDict[sameValueCount].append(comparision_pair)

        # List that keeps importance of the issues in descending order.
        issuesOrderings = [x for x in range(len(self.utility_space.issue_names))]

        for sameValueCount, comparison_pair_list in comparablesDict.items():
            # If the same value number is 1, there is no information that we can gain.
            if sameValueCount == 1:
                continue

            for comparison_pair in comparison_pair_list:
                comparingIssuesCount = len(comparison_pair.comparingIssues)
                conftlictingIssues = []

                for iss in range(comparingIssuesCount):
                    first_value = comparison_pair.first_offer.get_bid()[iss]
                    second_value = comparison_pair.second_offer.get_bid()[iss]

                    if first_value < second_value:
                        conftlictingIssues.append(iss)

                if len(conftlictingIssues) > 0:
                    nonConflictingIssues = [
                        x
                        for x in range(comparingIssuesCount)
                        if x not in conftlictingIssues
                    ]

                    for q in conftlictingIssues:
                        for z in nonConflictingIssues:
                            conflict = comparison_pair.comparingIssues[q]
                            nonConflict = comparison_pair.comparingIssues[z]

                            if issuesOrderings.index(
                                nonConflict
                            ) < issuesOrderings.index(conflict):
                                (
                                    issuesOrderings[nonConflict],
                                    issuesOrderings[conflict],
                                ) = (
                                    issuesOrderings[conflict],
                                    issuesOrderings[nonConflict],
                                )

        issueWeights = [1.0 / len(self.utility_space.issue_names)] * len(
            self.utility_space.issue_names
        )

        mid = int(len(self.utility_space.issue_names) / 2.0)
        diff = 1.0 / (2 ** (len(self.utility_space.issue_names) - 1))
        issueWeights[mid - 1] -= diff / 2.0
        issueWeights[mid] += diff / 2.0
        issueWeights[mid - 2] = issueWeights[mid - 1] - diff
        issueWeights[mid + 1] = issueWeights[mid] + diff

        estimatedOpponentPreference = copy.deepcopy(self.utility_space)

        for issue_index, issue_name in enumerate(
            estimatedOpponentPreference.issue_names
        ):
            estimatedOpponentPreference.issue_weights[issue_index] = issueWeights[
                issue_index
            ]

        realOpponentPreference = UtilitySpace("HATN/Domain/Fruit/Fruits_A_Human.xml")

        # print("estimated weights:", estimatedOpponentPreference.issue_weights)
        # print("real weights:", realOpponentPreference.issue_weights)
        #
        #
        # print("Issue orderings:", issuesOrderings)
        #
        # ### RMSE ###
        # sum = 0
        # for opponentOffer in opponentOfferHistory:
        # 	#print("real utility and x:", realOpponentPreference.get_offer_utility(opponentOffer), opponentOffer.get_bid())
        # 	sum += (estimatedOpponentPreference.get_offer_utility(opponentOffer) - realOpponentPreference.get_offer_utility(opponentOffer)) ** 2
        #
        # print("estimated utility and x:", estimatedOpponentPreference.get_offer_utility(opponentOfferHistory[-1].get_bid()), opponentOfferHistory[-1].get_bid())
        # print("real utility and x:", realOpponentPreference.get_offer_utility(opponentOfferHistory[-1]), opponentOfferHistory[-1].get_bid())
        #
        # rmse_a = sum / len(opponentOfferHistory)
        # rmse_b = sqrt(rmse_a)
        #
        # print("RMSE: ", rmse_b)

        return estimatedOpponentPreference
