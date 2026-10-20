from HATN import negoAction


class ComparisonObject:
    def __init__(self, first_offer, second_offer):
        self.comparingIssues = []
        self.issue_size = len(first_offer.get_bid())
        self.first_offer, self.second_offer = self.reduce_elements(
            first_offer, second_offer
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.first_offer) + ">" + str(self.second_offer)

    def __eq__(self, other):
        """
        Takes comparison object as input as returns true if they are same otherwise false.
        """
        return (
            self.first_offer == other.first_offer
            and self.second_offer == other.second_offer
        )

    def reduce_elements(self, first_offer, second_offer):  # fix this with lambdas
        a = []
        b = []
        for i in range(self.issue_size):
            if first_offer.get_bid()[i] != second_offer.get_bid()[i]:
                a.append(first_offer.get_bid()[i])
                b.append(second_offer.get_bid()[i])
                self.comparingIssues.append(i)
        if len(a) == 0:
            return first_offer, second_offer
        return negoAction.Offer(a), negoAction.Offer(b)

    def isComparable(self, other):
        return self.comparingIssues in other.comparingIssues
