from flask import Flask, render_template, request
import numpy as np
from skfuzzy.control import Antecedent, Consequent, Rule, ControlSystem, ControlSystemSimulation
from skfuzzy.membership import trapmf, trimf
import tensorflow as tf
from train_data import train_data, train_labels

app = Flask(__name__)

def fuzzy_logic(temperatur_input, kelembapan_input, konsentrasigas_input, intensitascahaya_input, api_input):
    # Definisikan variledlampel fuzzy untuk 5 input sensor
    temperatur = Antecedent(np.arange(0, 51, 1), 'temperatur')  # Rentang temperatur 0-43
    kelembapan = Antecedent(np.arange(0, 101, 1), 'kelembapan')  # Rentang kelembapan 0-101
    konsentrasigas = Antecedent(np.arange(0, 601, 1), 'konsentrasigas')  # Rentang  0-11
    intensitascahaya = Antecedent(np.arange(0, 501, 1), 'intensitascahaya')  # Rentang  0-51
    api = Antecedent(np.arange(0, 2, 1), 'api')  # Rentang api 0-1001

    # Output untuk 5 aktuator
    ac = Consequent(np.arange(0, 11, 1), 'ac')  # Rentang ac 0-10
    humidifier = Consequent(np.arange(0, 11, 1), 'humidifier')  # Rentang humidifier 0-10
    exhaustfan = Consequent(np.arange(0, 11, 1), 'exhaustfan')  # Rentang exhaustfan 0-10
    ledlamp = Consequent(np.arange(0, 11, 1), 'ledlamp')  # Rentang ledlamp 0-10
    buzzer = Consequent(np.arange(0, 2, 1), 'buzzer')  # Rentang buzzer 0-10

    # Fungsi keanggotaan untuk input (low, medium, high)
    # Keanggotaan untuk temperatur
    temperatur['low'] = trimf(temperatur.universe, [0, 5, 25])  # Low antara 0 dan 14
    temperatur['medium'] = trimf(temperatur.universe, [20, 25, 30])  # Medium antara 10 dan 32
    temperatur['high'] = trimf(temperatur.universe, [35, 50, 50])  # High antara 28 dan 43

    # Keanggotaan untuk kelembapan
    kelembapan['low'] = trimf(kelembapan.universe, [0, 0, 50])  # Low antara 0 dan 34
    kelembapan['medium'] = trimf(kelembapan.universe, [40, 50, 80])  # Medium antara 30 dan 70
    kelembapan['high'] = trimf(kelembapan.universe, [70, 100, 100])  # High antara 60 dan 101

    # Keanggotaan untuk konsentrasi gas
    konsentrasigas['low'] = trimf(konsentrasigas.universe, [0, 0, 200])  # Low antara 0 dan 3
    konsentrasigas['medium'] = trimf(konsentrasigas.universe, [150, 300, 450])  # Medium antara 2 dan 8
    konsentrasigas['high'] = trimf(konsentrasigas.universe, [400, 600, 600])  # High antara 7 dan 11

    # Keanggotaan untuk intensitas cahaya
    intensitascahaya['low'] = trimf(intensitascahaya.universe, [0, 0, 200])  # Low antara 0 dan 17
    intensitascahaya['medium'] = trimf(intensitascahaya.universe, [150, 250, 350])  # Medium antara 10 dan 40
    intensitascahaya['high'] = trimf(intensitascahaya.universe, [300, 500, 500])  # High antara 30 dan 51

    # Keanggotaan untuk api
    api['notdetected'] = trimf(api.universe, [0, 0, 0.5])  # On antara 0 dan 300
    api['detected'] = trimf(api.universe, [0.5, 1, 1])  # Off antara 200 dan 600

    # Fungsi keanggotaan untuk output (ac, humidifier, exhaustfan, ledlamp, buzzer)
    ac['low'] = trimf(ac.universe, [0, 0, 5])  # Low antara 0 dan 3
    ac['medium'] = trimf(ac.universe, [3, 5, 8])  # medium antara 2 dan 8
    ac['high'] = trimf(ac.universe, [7, 10, 10])  # high antara 7 dan 10

    humidifier['low'] = trimf(humidifier.universe, [0, 0, 5])  # low antara 0 dan 3
    humidifier['medium'] = trimf(humidifier.universe, [3, 5, 8])  # medium antara 2 dan 8
    humidifier['high'] = trimf(humidifier.universe, [7, 10, 10])  # high antara 7 dan 10

    exhaustfan['low'] = trimf(exhaustfan.universe, [0, 0, 5])  # low antara 0 dan 3
    exhaustfan['medium'] = trimf(exhaustfan.universe, [3, 5, 8])  # medium antara 2 dan 8
    exhaustfan['high'] = trimf(exhaustfan.universe, [7, 10, 10])  # high antara 7 dan 10

    ledlamp['low'] = trimf(ledlamp.universe, [0, 0, 5])  # low antara 0 dan 3
    ledlamp['medium'] = trimf(ledlamp.universe, [3, 5, 8])  # medium antara 2 dan 8
    ledlamp['high'] = trimf(ledlamp.universe, [7, 10, 10])  # high antara 7 dan 11  

    buzzer['on'] = trimf(buzzer.universe, [0, 0, 0.5])  # low antara 0 dan 17
    buzzer['off'] = trimf(buzzer.universe, [0.5, 1, 1])  # medium antara 10 dan 40

    # Aturan fuzzy untuk output ac, humidifier, exhaustfan, ledlamp, dan buzzer
    rule1 = Rule(temperatur['low'], ac['low'])
    rule2 = Rule(temperatur['medium'], ac['medium'])
    rule3 = Rule(temperatur['high'], ac['high'])

    rule4 = Rule(kelembapan['low'], humidifier['low'])
    rule5 = Rule(kelembapan['medium'], humidifier['medium'])
    rule6 = Rule(kelembapan['high'], humidifier['high'])

    rule7 = Rule(konsentrasigas['low'], exhaustfan['low'])
    rule8 = Rule(konsentrasigas['medium'], exhaustfan['medium'])
    rule9 = Rule(konsentrasigas['high'], exhaustfan['high'])

    rule10 = Rule(intensitascahaya['low'], ledlamp['low'])
    rule11 = Rule(intensitascahaya['medium'], ledlamp['medium'])
    rule12 = Rule(intensitascahaya['high'], ledlamp['high'])

    rule13 = Rule(api['notdetected'], buzzer['on'])
    rule14= Rule(api['detected'], buzzer['off'])

    # Sistem kontrol fuzzy
    ac_ctrl = ControlSystem([rule1, rule2, rule3])
    humidifier_ctrl = ControlSystem([rule4, rule5, rule6])
    exhaustfan_ctrl = ControlSystem([rule7, rule8, rule9])
    ledlamp_ctrl = ControlSystem([rule10, rule11, rule12])
    buzzer_ctrl = ControlSystem([rule13, rule14])

    ac_simulation = ControlSystemSimulation(ac_ctrl)
    humidifier_simulation = ControlSystemSimulation(humidifier_ctrl)
    exhaustfan_simulation = ControlSystemSimulation(exhaustfan_ctrl)
    ledlamp_simulation = ControlSystemSimulation(ledlamp_ctrl)
    buzzer_simulation = ControlSystemSimulation(buzzer_ctrl)

    # Input untuk sensor
    try:
        # debug ac
        print("AC")
        ac_simulation.input['temperatur'] = temperatur_input
        print(ac_simulation.input)

        # debug humidifier
        print("HUMIDIFIER")
        humidifier_simulation.input['kelembapan'] = kelembapan_input
        print(humidifier_simulation.input)

        # debug exhaustfan
        print("EXHAUST Fan")
        exhaustfan_simulation.input['konsentrasigas'] = konsentrasigas_input
        print(exhaustfan_simulation.input)

        # debug ledlamp
        print("LED Lamp")
        ledlamp_simulation.input['intensitascahaya'] = intensitascahaya_input
        print(ledlamp_simulation.input)

        # debug buzzer
        print("BUZZER")
        buzzer_simulation.input['api'] = api_input
        print(buzzer_simulation.input)

        # Menjalankan simulasi
        ac_simulation.compute()
        humidifier_simulation.compute()
        exhaustfan_simulation.compute()
        ledlamp_simulation.compute()
        buzzer_simulation.compute()
        print("DONE COMPUTE")
        print(ac_simulation.output)
        print(humidifier_simulation.output)
        print(exhaustfan_simulation.output)
        print(ledlamp_simulation.output)
        print(buzzer_simulation.output)

        # Output yang dihitung
        return ac_simulation.output['ac'], humidifier_simulation.output['humidifier'], exhaustfan_simulation.output['exhaustfan'], ledlamp_simulation.output['ledlamp'], buzzer_simulation.output['buzzer']
    except Exception as e:
        print(f"Error: {e}")  # Cetak error untuk debugging
        return 0, 0, 0, 0, 0

# Neural Network untuk prediksi output (contoh sederhana)
def neural_network_predict(temperatur, kelembapan, konsentrasigas, intensitascahaya, api, train_data, train_labels, epochs=100):
    # Model Neural Network untuk prediksi output
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_dim=5),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5)  # Output 5 Aktuator
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Training model
    model.fit(train_data, train_labels, epochs=epochs, verbose=1)

    # Input untuk prediksi
    input_data = np.array([[temperatur, kelembapan, konsentrasigas, intensitascahaya, api]])

    # Prediksi output
    output = model.predict(input_data)
    return output[0][0], output[0][1], output[0][2], output[0][3], output[0][4]

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html',  ac=0, humidifier=0, exhaustfan=0, ledlamp=0, buzzer=0,
                               nn_ac=0, nn_humidifier=0, nn_exhaustfan=0, nn_ledlamp=0, nn_buzzer=0)

# Proses data dari frontend
@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        temperatur = float(request.form['temperatur'])
        kelembapan = float(request.form['kelembapan'])
        konsentrasigas = float(request.form['konsentrasigas'])
        intensitascahaya = float(request.form['intensitascahaya'])
        api = float(request.form['api'])

        print("====================================")
        print("Data Input")
        print("\ntemperatur:", temperatur, "\nkelembapan:", kelembapan, "\nkonsentrasigas:", konsentrasigas, "\nintensitascahaya:", intensitascahaya, "\napi:", api)
        print("====================================")

        # Proses dengan Fuzzy Logic
        print("Fuzzy Logic")
        ac_output, humidifier_output, exhaustfan_output, ledlamp_output, buzzer_output = fuzzy_logic(temperatur, kelembapan, konsentrasigas, intensitascahaya, api)

        # Proses dengan Neural Network
        print("Neural Network")
        nn_ac, nn_humidifier, nn_exhaustfan, nn_ledlamp, nn_buzzer = neural_network_predict(temperatur, kelembapan, konsentrasigas, intensitascahaya, api, train_data, train_labels)

        # debug print
        print("====================================")
        print(f"AC: {ac_output}")
        print(f"Humidifier: {humidifier_output}")
        print(f"Exhaust Fan: {exhaustfan_output}")
        print(f"Led Lamp: {ledlamp_output}")
        print(f"Buzzer: {buzzer_output}")
        print(f"NN AC: {nn_ac}")
        print(f"NN Humidifier: {nn_humidifier}")
        print(f"NN Exhaust Fan: {nn_exhaustfan}")
        print(f"NN Led Lamp: {nn_ledlamp}")
        print(f"NN Buzzer: {nn_buzzer}")


        # Mengirim hasil ke frontend
        return render_template('index.html', ac=ac_output, humidifier=humidifier_output, exhaustfan=exhaustfan_output, ledlamp=ledlamp_output, buzzer=buzzer_output,
                               nn_ac=nn_ac, nn_humidifier=nn_humidifier, nn_exhaustfan=nn_exhaustfan, nn_ledlamp=nn_ledlamp, nn_buzzer=nn_buzzer)

if __name__ == '__main__':
    app.run(debug=True)
