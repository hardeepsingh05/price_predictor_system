from abc import ABC, abstractmethod

# Step 01: Define the Product Interface
class Coffee(ABC):
    @abstractmethod
    def make_it(self):
        pass

# Step 02: Implement Concrete products
class Espresso(Coffee):
    def make_it(Coffee):
        return "Preparing a rich and strong Espresso"
    
class Latte(Coffee):
    def make_it(Coffee):
        return "Preparing a smooth and creamy Latte"
    
class Cappuccino(Coffee):
    def make_it(Coffee):
        return "Preparing a frothy Cappuccino"
    

# Step 03: Implement the Factory (CoffeeMachine)
class CoffeeMachine:
    def Make_Coffee(self,coffee_type):
        if coffee_type == "Espresso":
            return Espresso().make_it()
        elif coffee_type  == "Latte":
            return Latte().make_it()
        elif coffee_type == "Cappuccino":
            return Cappuccino().make_it()
        else:
            return "Plase enter correct coffee type!!"

# Step 04: Use the Factory to create Products
if __name__ == "__main__":
    machine = CoffeeMachine()   # define a object for defined class

    coffee = machine.Make_Coffee("Espresso")
    print(coffee)

    coffee = machine.Make_Coffee("Chaa")
    print(coffee)
