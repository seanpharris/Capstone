import sys
import tkinter as tk
import tkinter.font as tkFont

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.widgets import Slider, Button
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#  pulls data from csv file
loaded_csv = pd.read_csv('data files/Housing.csv')

#  Split data into feature_list and target_list
#  feature_list is used to train on to ensure accurate prediction of price
feature_list = loaded_csv[['Average Area Income', 'Average Area House Age', 'Average Area Number of Rooms',
                           'Average Area Number of Bedrooms', 'Area Population']]

#  target_list holds the prices which is what we hope the data predicts
target_list = loaded_csv['Price']

#  this splits the data for a testing model and a training model
#  the splits is 70% - training/30% - testing
feature_list_train, feature_list_test, target_list_train, target_list_test = train_test_split(feature_list, target_list,
                                                                                              test_size=0.3,
                                                                                              random_state=101)

#  the algo used is linear regression and we will use the following line provided by the sklearn library
lm = LinearRegression()

#  the data is then fit into the algo
lm.fit(feature_list_train, target_list_train)

#  to evaluate the model, we get the coefficients
coeff_df = pd.DataFrame(lm.coef_, feature_list.columns, columns=['Coefficient'])

#  *******Needs to be placed in GUI******  the following line will print the coefficients
# print(coeff_df)

#  predictions from the feature list are made using the linear regression algo
age = loaded_csv['Average Area House Age']
rooms = loaded_csv['Average Area Number of Rooms']
bedrooms = loaded_csv['Average Area Number of Bedrooms']

predictions = lm.predict(feature_list_test)


class App:

    def __init__(self, root):
        # setting title
        root.title("ProjectMyMarket")
        # setting window size
        width = 950
        height = 500
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        straighten = '%dx%d+%d+%d' % (width, height, (screen_width - width) / 2, (screen_height - height) / 2)
        root.geometry(straighten)
        root["bg"] = "#7CFFA4"
        root.resizable(width=False, height=False)

        #  exit button
        exit_button = tk.Button(root)
        exit_button["bg"] = "#efefef"
        font = tkFont.Font(family='Times', size=10)
        exit_button["font"] = font
        exit_button["fg"] = "#000000"
        exit_button["justify"] = "center"
        exit_button["text"] = "Exit"
        exit_button.place(x=800, y=470, width=70, height=25)
        exit_button["command"] = self.exit_button_command

        #  prediction button
        prediction_button = tk.Button(root)
        prediction_button["bg"] = "#efefef"
        font = tkFont.Font(family='Times', size=10)
        prediction_button["font"] = font
        prediction_button["fg"] = "#000000"
        prediction_button["justify"] = "center"
        prediction_button["text"] = "Predict market"
        prediction_button.place(x=600, y=470, width=100, height=25)
        prediction_button["command"] = self.prediction_command

        #  Button for scatter plot
        scatter_plot_button = tk.Button(root)
        scatter_plot_button["bg"] = "#efefef"
        font = tkFont.Font(family='Times', size=10)
        scatter_plot_button["font"] = font
        scatter_plot_button["fg"] = "#000000"
        scatter_plot_button["justify"] = "center"
        scatter_plot_button["text"] = "View data as scatter plot"
        scatter_plot_button.place(x=280, y=350, width=150, height=47)
        scatter_plot_button["command"] = self.scatter_plot_button_command

        #  Button for heat map
        heat_map_button = tk.Button(root)
        heat_map_button["bg"] = "#efefef"
        font = tkFont.Font(family='Times', size=10)
        heat_map_button["font"] = font
        heat_map_button["fg"] = "#000000"
        heat_map_button["justify"] = "center"
        heat_map_button["text"] = "View data in heat map"
        heat_map_button.place(x=440, y=400, width=150, height=47)
        heat_map_button["command"] = self.heat_map_button_command

        # button to display graphical statistics of data (pair plots)
        pair_plot_button = tk.Button(root)
        pair_plot_button["bg"] = "#efefef"
        font = tkFont.Font(family='Times', size=10)
        pair_plot_button["font"] = font
        pair_plot_button["fg"] = "#000000"
        pair_plot_button["justify"] = "center"
        pair_plot_button["text"] = "View data as pair plots"
        pair_plot_button.place(x=440, y=350, width=150, height=48)
        pair_plot_button["command"] = self.pair_plot_button_command

        #  button to display histogram
        histogram_button = tk.Button(root)
        histogram_button["bg"] = "#efefef"
        font = tkFont.Font(family='Times', size=10)
        histogram_button["font"] = font
        histogram_button["fg"] = "#000000"
        histogram_button["justify"] = "center"
        histogram_button["text"] = "View data as histogram plot"
        histogram_button.place(x=600, y=350, width=170, height=48)
        histogram_button["command"] = self.histogram_button_command

        #  message area to show sample of data
        sample_data_message = tk.Message(root)
        font = tkFont.Font(family='Times', size=12)
        sample_data_message["font"] = font
        sample_data_message["fg"] = "#333333"
        sample_data_message["bg"] = "#CDCDCD"
        pd.set_option('display.max_colwidth', 0)
        sample_data_message.place(x=70, y=10, width=828, height=298)

        label_da = tk.Label(root, text=loaded_csv.describe())
        label_da.place(x=70, y=30, width=828, height=270)

        label_pred = tk.Label(sample_data_message, text="Data statistics/totals")
        label_pred.place(rely=0, relx=0)

    #  command to exit application
    def exit_button_command(self):
        sys.exit()

    #  the following is a command for a button that displays a scatter plot of the predictions made from the features
    def scatter_plot_button_command(self):
        plt.scatter(target_list_test, predictions)
        plt.show()

    #  command for button to display heat map
    def heat_map_button_command(self):
        sns.heatmap(loaded_csv.corr(), annot=True)
        plt.show()

    #  the following is command to display graphical statistics of the data
    def pair_plot_button_command(self):
        sns.pairplot(loaded_csv)
        plt.show()

    #  the following is a command for a button to display a histogram of the predictions
    def histogram_button_command(self):
        sns.distplot((target_list_test - predictions), bins=50)
        plt.show()

    def prediction_command(self):
        # Create a subplot
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)
        r = 0.6
        g = 0.2
        b = 0.5

        # Bar heights - numerical
        algo = 300000
        low = target_list.min(skipna=True)
        high = target_list.max(skipna=True)

        # Create and plot a bar chart
        year = ['Lowest', 'Predicted', 'Highest']
        # Bar heights entered into list
        production = [low, algo, high]
        plt.bar(year, production, edgecolor="black")

        # Position for sliders
        axage = plt.axes([0.25, 0.2, 0.65, 0.03])
        axrooms = plt.axes([0.25, 0.15, 0.65, 0.03])
        axbedrooms = plt.axes([0.25, 0.1, 0.65, 0.03])

        # Create a slider from 0.0 to 10.0 in axes axage
        # 5 as initial value.
        red = Slider(axage, 'age of house', 0.0, 10.0, 5)

        # Create a slider from 0.0 to 10.0 in axes axrooms
        # 5 as initial value.
        green = Slider(axrooms, 'rooms', 0.0, 10.0, 5)

        # Create a slider from 0.0 to 10.0 in axes axbedrooms
        # 5 as initial value
        blue = Slider(axbedrooms, 'bed rooms', 0.0, 10.0, 5)

        # Create function to be called when slider value is changed

        ax.bar(year, production, edgecolor="black")

        def update(event):
            x = rooms.mean(skipna=True) * bedrooms.mean(skipna=True) * age.mean()
            y = target_list.mean(skipna=True) / x
            r = red.val
            g = green.val
            b = blue.val
            z = r * g * b
            price = y * z
            print(price)
            algo = price
            low = target_list.min(skipna=True)
            high = target_list.max(skipna=True)
            # Create and plot a bar chart
            year = ['Lowest', 'Predicted', 'Highest']
            # Bar heights entered into list
            production = [low, algo, high]
            ax.bar(year, production, edgecolor="black")

            r2_score = lm.score(feature_list_test, target_list_test)
            score = r2_score*100
            textstr = ('%s percent prediction accuracy' % score)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top')

        c_button = plt.axes([0.5, 0.025, 0.1, 0.04])
        calc = Button(c_button, 'Calculate', color='gold',
                      hovercolor='skyblue')
        # Call resetSlider function when clicked on reset button
        calc.on_clicked(update)
        # Display graph
        plt.show()


#  begin applications and calls for the following
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
