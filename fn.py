import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
import pickle
from feature_setting import FeatureExtraction
import pandas as pd
import numpy as np
from kivy.core.window import Window
import warnings


#Window.clearcolor = 

class CheckURL(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.label1 = Label(text = "COPY AND PASTE ANY URL NOW")
        self.url_input = TextInput(text='Enter URL')
        self.result_label = Label(text='')
        self.prob_1 = Label(text = "")
        self.prob_2 = Label(text = "")
        button = Button(text='Check URL', on_press=self.check_url)
        layout.add_widget(self.label1)
        layout.add_widget(self.url_input)
        layout.add_widget(button)
        layout.add_widget(self.result_label)
        layout.add_widget(self.prob_1)
        layout.add_widget(self.prob_2)

        return layout

    def check_url(self, instance):

        file = open("model_psh.pkl","rb")
        gbc = pickle.load(file)
        file.close()
        url = self.url_input.text

        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 
        d = pd.DataFrame(x)
        y_pred = gbc.predict(x)[0]

        if y_pred == 1:
            self.result_label.text = "The Site is Safe"
        else:
            self.result_label.text = "The Sitte is Unsafe"

        y_yes = round(gbc.predict_proba(x)[0,0],4)
        y_no = round(gbc.predict_proba(x)[0,1],4)
        a_res = str(y_yes)
        b_res = str(y_no)
        self.prob_1.text = "Safe Pobability  =  " + a_res
        self.prob_2.text = "Unsafe Probability =  " + b_res


if __name__ == '__main__':
    CheckURL().run()