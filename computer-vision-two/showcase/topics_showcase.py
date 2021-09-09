from demo_colour_spaces import demo_colour_spaces



topics = {
    1:"Colour Spaces",
    2:"Image Annotations",
    3:"Colour Histograms",
    }


def MainMenu():
    # DisplayOptions()
    selection = get_user_choice()
    demo_topic(selection)

    return
    



def get_user_choice():
    

    for key in topics:
        print(f"{key}: {topics[key]}")

    selection = int(input("Please select an option: "))
    print(topics[selection])


    return selection

def demo_topic(selection):
    print(f">>>{topics[selection]} Demo")

    if selection == 1:
        demo_colour_spaces()
    elif selection == 2:
        # Image annotations demo
        pass

    return




if __name__ != "__main__":
    raise Exception("I should not be imported, please run topics_showcase.py directly.")

if __name__ == "__main__":
    # Display a main menu
    MainMenu()
