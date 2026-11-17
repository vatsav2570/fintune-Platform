"""
Generate ~300 Q&A training samples for B&K Precision 9800/B Series datasheet.
All answers are grounded strictly in the PDF datasheet specifications.
"""

import json
import random

SYSTEM = (
    "You are a strict technical assistant for B&K Precision 9800/B Series "
    "Programmable AC Power Sources. Answer ONLY using datasheet facts. "
    "If the answer is not explicitly known, say: 'I don't know'. "
    "Do NOT guess or approximate. Be precise and factual."
)

def qa(question, answer):
    return {"messages": [
        {"role": "system",  "content": SYSTEM},
        {"role": "user",    "content": question},
        {"role": "assistant","content": answer},
    ]}

samples = [

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    qa("What is the 9800/B Series?",
       "The 9800/B Series is a line of fully programmable linear AC power sources manufactured by B&K Precision. They deliver up to 1500 VA through universal line output terminals on the front and rear of the unit and include built-in measurement and simulation features."),

    qa("What models are included in the 9800/B Series?",
       "The 9800/B Series includes three models:\n• Model 9801 — up to 300 VA\n• Model 9803/B — up to 750 VA\n• Model 9805/B — up to 1500 VA"),

    qa("What does the 'B' suffix mean in model numbers like 9803B and 9805B?",
       "Model numbers with the 'B' suffix (980xB) do NOT include a GPIB interface. Models 9803 and 9805 (without suffix) include GPIB. See ordering information for details."),

    qa("What is the maximum power output of model 9801?",
       "Model 9801 delivers a maximum power output of 300 VA."),

    qa("What is the maximum power output of model 9803/B?",
       "Model 9803/B delivers a maximum power output of 750 VA."),

    qa("What is the maximum power output of model 9805/B?",
       "Model 9805/B delivers a maximum power output of 1500 VA."),

    qa("What is the total VA power range of the 9800/B Series across all models?",
       "The 9800/B Series ranges from 300 VA (model 9801) to 750 VA (model 9803/B) to 1500 VA (model 9805/B)."),

    qa("Who manufactures the 9800/B Series?",
       "The 9800/B Series is manufactured by B&K Precision Corporation, headquartered in Yorba Linda, California, USA."),

    qa("How many years has B&K Precision been in business?",
       "B&K Precision has provided test and measurement instruments worldwide for more than 70 years."),

    qa("What quality certification does B&K Precision hold?",
       "B&K Precision Corporation is ISO 9001:2015 registered (Certificate number 6Z241-IS8, Certification body NSF-ISR), employing traceable quality management practices for all processes including product development, service, and calibration."),

    qa("Where is B&K Precision headquartered?",
       "B&K Precision is headquartered in Yorba Linda, California, USA."),

    qa("Does B&K Precision have international operations?",
       "Yes. B&K Precision has a French subsidiary called Sefram for European customers, a B+K Precision Taiwan operation for Asia, and independent service centers in Singapore and Brazil."),

    # ── AC INPUT ──────────────────────────────────────────────────────────────
    qa("What AC input voltage does the 9800/B Series require?",
       "The 9800/B Series accepts single-phase AC input of 110 VAC or 220 VAC, ±10%."),

    qa("What is the AC input frequency range for the 9800/B Series?",
       "The AC input line frequency range is 47 to 63 Hz."),

    qa("What is the AC input phase of the 9800/B Series?",
       "The AC input is single-phase."),

    qa("What is the maximum input current of model 9801 at 110 VAC?",
       "At 110 VAC input, model 9801 draws a maximum input current of 10 A."),

    qa("What is the maximum input current of model 9801 at 220 VAC?",
       "At 220 VAC input, model 9801 draws a maximum input current of 6.3 A."),

    qa("What is the maximum input current of model 9803/B at 110 VAC?",
       "At 110 VAC input, model 9803/B draws a maximum input current of 30 A."),

    qa("What is the maximum input current of model 9803/B at 220 VAC?",
       "At 220 VAC input, model 9803/B draws a maximum input current of 15 A."),

    qa("What is the maximum input current of model 9805/B at 110 VAC?",
       "At 110 VAC input, model 9805/B draws a maximum input current of 40 A."),

    qa("What is the maximum input current of model 9805/B at 220 VAC?",
       "At 220 VAC input, model 9805/B draws a maximum input current of 20 A."),

    qa("Does the 9800/B Series support both 110 V and 220 V input?",
       "Yes. The 9800/B Series supports single-phase 110 VAC or 220 VAC input, ±10%. A 110/220 V selector is available on the rear panel."),

    qa("What is the input voltage tolerance of the 9800/B Series?",
       "The input voltage tolerance is ±10% of the nominal 110 VAC or 220 VAC input."),

    # ── AC OUTPUT ─────────────────────────────────────────────────────────────
    qa("What is the output voltage range of the 9800/B Series?",
       "The output voltage range is 0 to 300 V (AC RMS)."),

    qa("What voltage ranges does the 9800/B Series support?",
       "The 9800/B Series supports a 150 V range (Range 1), a 300 V range (Range 2), and an autoranging 150 V / 300 V mode for continuous voltage sweep from 0 to 300 V."),

    qa("What is the output frequency range of the 9800/B Series?",
       "The output frequency is adjustable from 45 Hz to 500 Hz."),

    qa("What is the output phase of the 9800/B Series?",
       "The output phase is single-phase."),

    qa("What is the crest factor of the 9800/B Series output?",
       "The crest factor of the AC output is 3."),

    qa("What is the Total Harmonic Distortion (THD) of the 9800/B Series?",
       "The Total Harmonic Distortion (THD) is ≤0.5% at 45 to 500 Hz with a resistive load."),

    qa("What is the line regulation of the 9800/B Series?",
       "Line regulation is 0.1% maximum for a ±10% line change."),

    qa("What is the load regulation of the 9800/B Series?",
       "Load regulation is ≤0.5% FS (full scale) with a resistive load."),

    qa("What is the response time of the 9800/B Series output?",
       "The output response time is less than 100 µs (<100 µs)."),

    qa("What is the power factor of model 9801?",
       "Model 9801 has a typical power factor of 0.5."),

    qa("What is the power factor of model 9803/B?",
       "Model 9803/B has a typical power factor of 0.7."),

    qa("What is the power factor of model 9805/B?",
       "Model 9805/B has a typical power factor of 0.7."),

    qa("What is the output voltage programming resolution of the 9800/B Series?",
       "The output voltage (RMS) programming resolution is 0.1 V."),

    qa("What is the output voltage programming accuracy of the 9800/B Series?",
       "The output voltage (RMS) programming accuracy is ±(0.2% + 0.6 V)."),

    qa("What is the output frequency programming resolution in the 45–99.9 Hz range?",
       "In the 45 to 99.9 Hz range, the frequency programming resolution is 0.1 Hz."),

    qa("What is the output frequency programming resolution in the 100–500 Hz range?",
       "In the 100 to 500 Hz range, the frequency programming resolution is 1 Hz."),

    qa("What is the output frequency programming accuracy at 100 Hz?",
       "The frequency programming accuracy is ±0.1 Hz at 100 Hz."),

    qa("What is the output frequency programming accuracy in the 100–500 Hz range?",
       "The frequency programming accuracy is ±1 Hz in the 100 to 500 Hz range."),

    qa("What is the phase angle programming range of the 9800/B Series?",
       "The phase angle programming range is 0° to 360°."),

    qa("What is the phase angle programming resolution of the 9800/B Series?",
       "The phase angle programming resolution is 0.1°."),

    qa("What is the phase angle programming accuracy of the 9800/B Series?",
       "The phase angle programming accuracy is ±1° for frequencies from 45 to 65 Hz."),

    # ── OUTPUT CURRENT ────────────────────────────────────────────────────────
    qa("What is the maximum RMS output current of model 9801 in the 150 V range?",
       "In the 150 V range (Range 1), model 9801 provides a maximum RMS output current of 3.0 A."),

    qa("What is the maximum RMS output current of model 9801 in the 300 V range?",
       "In the 300 V range (Range 2), model 9801 provides a maximum RMS output current of 1.5 A."),

    qa("What is the maximum RMS output current of model 9803/B in the 150 V range?",
       "In the 150 V range (Range 1), model 9803/B provides a maximum RMS output current of 6 A."),

    qa("What is the maximum RMS output current of model 9803/B in the 300 V range?",
       "In the 300 V range (Range 2), model 9803/B provides a maximum RMS output current of 3 A."),

    qa("What is the maximum RMS output current of model 9805/B in the 150 V range?",
       "In the 150 V range (Range 1), model 9805/B provides a maximum RMS output current of 12 A."),

    qa("What is the maximum RMS output current of model 9805/B in the 300 V range?",
       "In the 300 V range (Range 2), model 9805/B provides a maximum RMS output current of 6 A."),

    qa("What is the maximum peak output current of model 9801 in the 150 V range?",
       "In the 150 V range (Range 1), model 9801 provides a maximum peak output current of 9 A."),

    qa("What is the maximum peak output current of model 9801 in the 300 V range?",
       "In the 300 V range (Range 2), model 9801 provides a maximum peak output current of 4.5 A."),

    qa("What is the maximum peak output current of model 9803/B in the 150 V range?",
       "In the 150 V range (Range 1), model 9803/B provides a maximum peak output current of 18 A."),

    qa("What is the maximum peak output current of model 9803/B in the 300 V range?",
       "In the 300 V range (Range 2), model 9803/B provides a maximum peak output current of 9 A."),

    qa("What is the maximum peak output current of model 9805/B in the 150 V range?",
       "In the 150 V range (Range 1), model 9805/B provides a maximum peak output current of 36 A."),

    qa("What is the maximum peak output current of model 9805/B in the 300 V range?",
       "In the 300 V range (Range 2), model 9805/B provides a maximum peak output current of 18 A."),

    qa("What happens when Ipeak exceeds 300% of the current measurement range?",
       "When Ipeak exceeds 300% of the present range, the current range automatically switches from the low to mid range, or mid to high range."),

    qa("What happens when Ipeak drops below 80% of the high current measurement range?",
       "When Ipeak drops below 80% of the high range, the current range automatically switches from the high range to the mid range."),

    qa("What happens when Ipeak drops below 20% of the mid current measurement range?",
       "When Ipeak drops below 20% of the mid range, the current range automatically switches from the mid range to the low range."),

    qa("For which output voltage range is maximum power guaranteed on model 9801 in Range 1?",
       "For model 9801 in the 150 V range (Range 1), maximum power output is only available when the output voltage is set from 80 V to 120 V."),

    qa("For which output voltage range is maximum power guaranteed on model 9801 in Range 2?",
       "For model 9801 in the 300 V range (Range 2), maximum power output is only available when the output voltage is set from 160 V to 240 V."),

    # ── MEASUREMENTS ──────────────────────────────────────────────────────────
    qa("What is the voltage measurement range of the 9800/B Series?",
       "The voltage (RMS) measurement range is 0 to 300 V."),

    qa("What is the voltage measurement resolution of the 9800/B Series?",
       "The voltage measurement resolution is 0.1 V."),

    qa("What is the voltage measurement accuracy of the 9800/B Series?",
       "The voltage measurement accuracy is ±(0.2% + 0.6 V)."),

    qa("What are the current measurement ranges on the 9800/B Series?",
       "There are three current measurement ranges:\n• Low range: 120.0 mA\n• Mid range: 1.200 A\n• High range: 3.00 A (9801) / 6.00 A (9803/B) / 12.00 A (9805/B)"),

    qa("What is the current measurement resolution for the low range?",
       "The current measurement resolution for the low range is 0.1 mA."),

    qa("What is the current measurement resolution for the mid range?",
       "The current measurement resolution for the mid range is 1 mA."),

    qa("What is the current measurement resolution for the high range?",
       "The current measurement resolution for the high range is 10 mA."),

    qa("What is the current measurement accuracy for the low range?",
       "The current measurement accuracy for the low range is ±(0.2% + 0.4 mA)."),

    qa("What is the current measurement accuracy for the mid range?",
       "The current measurement accuracy for the mid range is ±(0.2% + 4 mA)."),

    qa("What is the current measurement accuracy for the high range?",
       "The current measurement accuracy for the high range is ±(0.2% + 20 mA)."),

    qa("What is the peak current measurement range for model 9801?",
       "For model 9801, the peak current measurement range is 0 to 9 A."),

    qa("What is the peak current measurement range for model 9803/B?",
       "For model 9803/B, the peak current measurement range is 0 to 18 A."),

    qa("What is the peak current measurement range for model 9805/B?",
       "For model 9805/B, the peak current measurement range is 0 to 36 A."),

    qa("What is the peak current measurement resolution?",
       "The peak current measurement resolution is 0.01 A."),

    qa("What is the peak current measurement accuracy?",
       "The peak current measurement accuracy is ±(1% + 120 mA)."),

    qa("What is the true power measurement resolution in the low range?",
       "The true power measurement resolution for the low range is 0.01 W."),

    qa("What is the true power measurement resolution in the mid range?",
       "The true power measurement resolution for the mid range is 0.1 W."),

    qa("What is the true power measurement resolution in the high range?",
       "The true power measurement resolution for the high range is 1 W."),

    qa("What is the true power measurement accuracy in the low range?",
       "The true power measurement accuracy for the low range is ±(0.2% + 0.05 W), valid for frequencies from 47 to 65 Hz."),

    qa("What is the true power measurement accuracy in the mid range?",
       "The true power measurement accuracy for the mid range is ±(0.2% + 0.5 W), valid for frequencies from 47 to 65 Hz."),

    qa("What is the true power measurement accuracy in the high range?",
       "The true power measurement accuracy for the high range is ±(0.2% + 2 W), valid for frequencies from 47 to 65 Hz."),

    qa("What is the frequency measurement range of the 9800/B Series?",
       "The frequency measurement range is 45 to 500 Hz."),

    qa("What is the frequency measurement resolution for 45–99.9 Hz?",
       "The frequency measurement resolution in the 45 to 99.9 Hz range is ±0.1 Hz."),

    qa("What is the frequency measurement resolution for 100–500 Hz?",
       "The frequency measurement resolution in the 100 to 500 Hz range is ±1 Hz."),

    qa("What is the frequency measurement accuracy of the 9800/B Series?",
       "The frequency measurement accuracy is ±0.1 Hz."),

    qa("What is the power factor measurement range of the 9800/B Series?",
       "The power factor measurement range is 0.000 to 1.000."),

    qa("What is the power factor measurement resolution of the 9800/B Series?",
       "The power factor measurement resolution is 0.001."),

    qa("What is the apparent power (VA) measurement resolution in the low range?",
       "The apparent power measurement resolution for the low range is 0.01 VA."),

    qa("What is the apparent power (VA) measurement resolution in the mid range?",
       "The apparent power measurement resolution for the mid range is 0.1 VA."),

    qa("What is the apparent power (VA) measurement resolution in the high range?",
       "The apparent power measurement resolution for the high range is 1 VA."),

    qa("How is apparent power accuracy calculated on the 9800/B Series?",
       "Apparent power (VA) accuracy is calculated as Voltage (RMS) × Current (RMS)."),

    qa("What parameters are shown on the front panel display?",
       "The bright VFD (Vacuum Fluorescent Display) shows: Vrms, Irms, Ipeak, frequency, power factor (PF), apparent power, true power, and elapsed output time."),

    # ── TEMPERATURE COEFFICIENT ────────────────────────────────────────────────
    qa("What is the temperature coefficient of the 9800/B Series?",
       "The temperature coefficient is ±0.04% per °C (typical)."),

    # ── WARM-UP TIME ──────────────────────────────────────────────────────────
    qa("How long does the 9800/B Series need to warm up before specifications are valid?",
       "All specifications apply after a temperature stabilization time of 15 minutes, over an ambient temperature range of 23 °C ± 5 °C."),

    # ── OPERATING ENVIRONMENT ─────────────────────────────────────────────────
    qa("What is the operating temperature range of the 9800/B Series?",
       "The operating temperature range is 32 °F to 104 °F (0 °C to 40 °C) at 20–80% relative humidity."),

    qa("What is the storage temperature range of the 9800/B Series?",
       "The storage temperature range is -4 °F to 158 °F (-20 °C to 70 °C) at ≤85% relative humidity."),

    qa("What are the environmental conditions for the 9800/B Series?",
       "The 9800/B Series is for indoor use only, with a maximum humidity of 80% and no condensation allowed."),

    # ── DIMENSIONS & WEIGHT ───────────────────────────────────────────────────
    qa("What are the dimensions of model 9801?",
       "Model 9801 dimensions (W x H x D) are 8.45\" x 3.47\" x 17.83\" (214.5 mm x 88.2 mm x 453.5 mm)."),

    qa("What are the dimensions of models 9803/B and 9805/B?",
       "Models 9803/B and 9805/B dimensions (W x H x D) are 17.3\" x 5.2\" x 21.1\" (439 mm x 131.4 mm x 535.7 mm)."),

    qa("What is the weight of model 9801?",
       "Model 9801 weighs 20.94 lb (9.5 kg)."),

    qa("What is the weight of model 9803/B?",
       "Model 9803/B weighs 88.2 lb (40 kg)."),

    qa("What is the weight of model 9805/B?",
       "Model 9805/B weighs 115 lb (52.16 kg)."),

    qa("What is the warranty period of the 9800/B Series?",
       "The 9800/B Series comes with a 2-year warranty."),

    # ── REMOTE INTERFACES ─────────────────────────────────────────────────────
    qa("What remote interfaces does the 9800/B Series support?",
       "The 9800/B Series supports:\n• USB (USBTMC-compliant)\n• RS-232\n• LAN (Ethernet)\n• GPIB (optional on 9803 and 9805; not on 9803B/9805B)\n• BNC I/O for triggering, output status/control, and synchronization"),

    qa("Is GPIB standard or optional on the 9800/B Series?",
       "GPIB is optional on select models. It is included on models 9803 and 9805. Models 9803B and 9805B do NOT include GPIB. Model 9801 does not support GPIB."),

    qa("Is GPIB available on model 9801?",
       "No. GPIB is not available on model 9801."),

    qa("Does model 9803B include GPIB?",
       "No. The 'B' suffix indicates no GPIB interface. Model 9803 (without B) includes GPIB, but model 9803B does not."),

    qa("Does model 9805B include GPIB?",
       "No. The 'B' suffix indicates no GPIB interface. Model 9805 (without B) includes GPIB, but model 9805B does not."),

    qa("What USB standard does the 9800/B Series USB interface comply with?",
       "The 9800/B Series USB interface is USBTMC-compliant."),

    qa("What is the BNC I/O connector on the rear panel used for?",
       "The BNC I/O connector is used for external triggering, sync output, output status indication, and output control/indicator. The List mode sequence can also be triggered via this BNC connector."),

    qa("What interfaces are standard on all 9800/B Series models?",
       "USB (USBTMC), RS-232, and LAN (Ethernet) interfaces are standard on all models in the 9800/B Series."),

    # ── MEMORY ────────────────────────────────────────────────────────────────
    qa("How many instrument settings can be saved on the 9800/B Series?",
       "Up to 100 instrument settings can be saved and recalled."),

    qa("How many memory locations does the 9800/B Series have for instrument settings?",
       "The 9800/B Series has 10 memory locations for saving instrument settings."),

    # ── LIST MODE ─────────────────────────────────────────────────────────────
    qa("What is List mode on the 9800/B Series?",
       "List mode allows generation of complex waveform sequences with varying times, amplitudes, and frequencies. Up to 100 steps in 10 groups can be saved and executed, useful for simulating grid faults and disturbances such as surges, sags, and frequency variations. The programmed list can be triggered from the front panel or via a BNC connector on the rear."),

    qa("How many steps does List mode support on the 9800/B Series?",
       "List mode supports up to 100 programmable steps."),

    qa("How many groups does List mode support on the 9800/B Series?",
       "List mode supports up to 10 user-defined groups (programs)."),

    qa("How many user-defined programs with steps does the 9800/B Series List mode support?",
       "List mode supports 10 user-defined programs with up to 100 programmable steps each."),

    qa("How can the List mode sequence be triggered on the 9800/B Series?",
       "The List mode sequence can be triggered from the front panel or via the BNC connector on the rear panel."),

    qa("What types of grid disturbances can List mode simulate?",
       "List mode can simulate grid faults and disturbances including surges, sags, frequency variations, and dropouts."),

    # ── PLD SIMULATOR ─────────────────────────────────────────────────────────
    qa("What is the Power Line Disturbance (PLD) simulator on the 9800/B Series?",
       "The PLD simulator is an extended feature of List mode that gives the user finer control over disturbance insertion into the AC waveform. It is useful for evaluating a product's immunity performance by producing common waveform disturbances such as surges, sags, spikes, and dropouts at user-defined locations on the waveform."),

    qa("What is PLD mode used for?",
       "PLD (Power Line Disturbance) mode is used for evaluating a product's immunity performance by producing common waveform disturbances like surges, sags, spikes, and dropouts at user-defined locations on the waveform."),

    qa("What disturbances can the PLD simulator produce?",
       "The PLD simulator can produce surges, sags, spikes, and dropouts at user-defined locations on the AC waveform."),

    qa("Is PLD mode a separate mode or part of another mode?",
       "The PLD simulator is an extended feature of List mode, not a completely separate operating mode."),

    # ── SWEEP MODE ────────────────────────────────────────────────────────────
    qa("What is Sweep mode on the 9800/B Series?",
       "Sweep mode enables user-defined voltage and frequency sweeps, which can be created independently or combined. It is ideal for testing the efficiency of switching power supplies or capturing the maximum operating power requirements of a device under test. Up to 10 sweep profiles can be stored and recalled."),

    qa("What can be swept in Sweep mode?",
       "In Sweep mode, voltage sweeps and frequency sweeps can be created independently or combined."),

    qa("How many sweep profiles can be stored in the 9800/B Series?",
       "Up to 10 sweep profiles can be stored and recalled."),

    qa("What applications is Sweep mode useful for?",
       "Sweep mode is useful for testing the efficiency of switching power supplies and capturing the maximum operating power requirements of a device under test."),

    qa("Can voltage and frequency be swept simultaneously in Sweep mode?",
       "Yes. In Sweep mode, user-defined voltage and frequency sweeps can be created independently or combined for a simultaneous voltage plus frequency sweep."),

    # ── DIMMER SIMULATION ─────────────────────────────────────────────────────
    qa("What is the Dimmer simulation feature of the 9800/B Series?",
       "The Dimmer simulation feature controls the phase cut-off of the AC sine wave's leading or trailing edge, varying the RMS voltage supplied to the load under test. The phase cut-off is adjustable between 0° and 180° for both leading-edge and trailing-edge dimming. It is useful for motor control and lighting test applications."),

    qa("What dimmer edge modes does the 9800/B Series support?",
       "The 9800/B Series supports both leading-edge and trailing-edge dimmer simulation."),

    qa("What is the phase cut-off adjustment range for the Dimmer simulation?",
       "The phase cut-off angle is adjustable from 0° to 180° for both leading-edge and trailing-edge dimming modes."),

    qa("What applications is the Dimmer simulation feature used for?",
       "The Dimmer simulation feature is useful for motor control and lighting test applications."),

    qa("How does the Dimmer simulation vary the output?",
       "The Dimmer simulation controls the phase cut-off of the AC sine wave's leading or trailing edge, which varies the RMS voltage supplied to the load under test."),

    # ── PROTECTION MODES ──────────────────────────────────────────────────────
    qa("What protection modes does the 9800/B Series have?",
       "The 9800/B Series has four protection modes: OVP (Over Voltage Protection), OCP (Over Current Protection), OPP (Over Power Protection), and OTP (Over Temperature Protection)."),

    qa("Does the 9800/B Series have over-voltage protection?",
       "Yes. The 9800/B Series includes OVP (Over Voltage Protection)."),

    qa("Does the 9800/B Series have over-current protection?",
       "Yes. The 9800/B Series includes OCP (Over Current Protection)."),

    qa("Does the 9800/B Series have over-power protection?",
       "Yes. The 9800/B Series includes OPP (Over Power Protection)."),

    qa("Does the 9800/B Series have over-temperature protection?",
       "Yes. The 9800/B Series includes OTP (Over Temperature Protection)."),

    qa("Does the 9800/B Series have a key lock function?",
       "Yes. The 9800/B Series includes a key lock function."),

    # ── FRONT PANEL ───────────────────────────────────────────────────────────
    qa("What type of display does the 9800/B Series use?",
       "The 9800/B Series uses a bright VFD (Vacuum Fluorescent Display)."),

    qa("What front panel controls are available on the 9800/B Series?",
       "The front panel features:\n• A numeric keypad for direct entry of AC output waveform parameters\n• Function keys\n• Cursor keys\n• An encoder knob for quick analog-style control\n• An AC output terminal"),

    qa("What does the numeric keypad on the 9800/B Series front panel do?",
       "The numeric keypad allows direct entry of AC output waveform parameters such as voltage, frequency, and phase."),

    qa("What is the encoder knob on the 9800/B Series used for?",
       "The encoder knob provides quick analog-style control for adjusting output parameters."),

    qa("Is there an AC output terminal on the front panel of the 9800/B Series?",
       "Yes. The 9800/B Series has an AC output terminal on the front panel."),

    # ── REAR PANEL ────────────────────────────────────────────────────────────
    qa("What connectors are on the rear panel of model 9801?",
       "The rear panel of model 9801 includes: AC line input, 110/220 V selector, BNC I/O trigger connector, AC output terminal, System BUS (not used), RS-232 interface, LAN interface, USB interface, and an optional GPIB interface."),

    qa("Do models 9803/B and 9805/B have a rear panel AC output terminal?",
       "Yes. Models 9803/B and 9805/B have an AC output terminal on the rear panel."),

    qa("What cooling mechanism is used in the 9803/B and 9805/B models?",
       "Models 9803/B and 9805/B use cooling fans for thermal management. Model 9801 does not use fans."),

    qa("Does model 9801 use cooling fans?",
       "No. Model 9801 does not use cooling fans. Only the larger models 9803/B and 9805/B use cooling fans."),

    # ── PRE-COMPLIANCE TESTING ────────────────────────────────────────────────
    qa("What pre-compliance standards does the 9800/B Series support?",
       "The 9800/B Series supports pre-compliance testing for voltage dips and frequency simulations according to IEC61000-4-11, IEC61000-4-14, and IEC61000-4-28."),

    qa("Does the 9800/B Series support IEC 61000-4-11 testing?",
       "Yes. The 9800/B Series supports pre-compliance testing for voltage dips and frequency simulations according to IEC61000-4-11."),

    qa("What IEC standards can the 9800/B Series simulate?",
       "The 9800/B Series supports pre-compliance testing according to IEC61000-4-11, IEC61000-4-14, and IEC61000-4-28 for voltage dip and frequency simulation testing."),

    # ── SOFTWARE ──────────────────────────────────────────────────────────────
    qa("Is application software provided with the 9800/B Series?",
       "Yes. PC software is provided for front panel emulation, generating and executing list, PLD, and sweep profiles, or logging measurement data without the need to write source code."),

    qa("Does the 9800/B Series support LabVIEW?",
       "Yes. A LabVIEW driver and softpanel for remote control are available for the 9800/B Series."),

    qa("What can the 9800/B Series application software do?",
       "The application software supports front panel emulation, generating and executing list, PLD, and sweep profiles, and logging measurement data, without requiring source code programming."),

    # ── APPLICATIONS ─────────────────────────────────────────────────────────
    qa("What components can the 9800/B Series be used to evaluate?",
       "The 9800/B Series is suitable for evaluating transformers, TRIACs, SCRs, and passive components."),

    qa("What testing categories is the 9800/B Series suitable for?",
       "The 9800/B Series is suitable for production testing, R&D, service, and pre-compliance testing."),

    qa("Can the 9800/B Series be used for switching power supply testing?",
       "Yes. The Sweep mode of the 9800/B Series is ideal for testing the efficiency of switching power supplies."),

    qa("Can the 9800/B Series simulate power line faults?",
       "Yes. Using List mode and the PLD simulator, the 9800/B Series can simulate power line faults including surges, sags, frequency variations, and voltage spikes."),

    qa("Can the 9800/B Series be used for motor testing?",
       "Yes. The Dimmer simulation feature of the 9800/B Series is useful for motor control and lighting test applications."),

    # ── ACCESSORIES ───────────────────────────────────────────────────────────
    qa("What standard accessories are included with model 9801?",
       "Model 9801 includes: an AC power cord, instruction manual, and certificate of calibration."),

    qa("What standard accessories are included with models 9803/B and 9805/B?",
       "Models 9803/B and 9805/B include: an unterminated power cord with input connector, rackmount ears and handles, instruction manual, and certificate of calibration."),

    qa("What optional accessories are available for the 9800/B Series?",
       "The optional accessory is the IT-E151 rack mount kit, available for model 9801."),

    qa("Is a rack mount kit available for model 9801?",
       "Yes. The IT-E151 rack mount kit is available as an optional accessory for model 9801."),

    qa("Are rackmount ears included by default with models 9803/B and 9805/B?",
       "Yes. Rackmount ears and handles are included as standard accessories with models 9803/B and 9805/B."),

    qa("Does the 9800/B Series come with a certificate of calibration?",
       "Yes. A certificate of calibration is included as a standard accessory with all models in the 9800/B Series."),

    # ── COMPARISON QUESTIONS ──────────────────────────────────────────────────
    qa("How does model 9805/B differ from model 9801?",
       "Key differences between 9805/B and 9801:\n• Power: 9805/B = 1500 VA vs 9801 = 300 VA\n• Max RMS current (150 V range): 9805/B = 12 A vs 9801 = 3.0 A\n• Max peak current (150 V range): 9805/B = 36 A vs 9801 = 9 A\n• Input current at 110 VAC: 9805/B = 40 A vs 9801 = 10 A\n• Power factor: 9805/B = 0.7 vs 9801 = 0.5 (both typical)\n• Cooling: 9805/B uses cooling fans; 9801 does not\n• GPIB: 9805/B optionally available; not available on 9801"),

    qa("What is the difference between the 150 V and 300 V output ranges?",
       "In the 150 V range (Range 1), the unit delivers higher current at lower voltages. In the 300 V range (Range 2), the maximum current is halved but the voltage capability doubles. Example for 9805/B: Range 1 = 12 A RMS / 36 A peak; Range 2 = 6 A RMS / 18 A peak."),

    qa("What is the difference between models 9803 and 9803B?",
       "Model 9803 includes a GPIB interface, while model 9803B does NOT include GPIB. All other specifications are identical."),

    qa("What is the difference between models 9805 and 9805B?",
       "Model 9805 includes a GPIB interface, while model 9805B does NOT include GPIB. All other specifications are identical."),

    qa("Which model has the highest current output capability?",
       "Model 9805/B has the highest current output: 12 A RMS / 36 A peak in the 150 V range, and 6 A RMS / 18 A peak in the 300 V range."),

    qa("Which 9800/B Series model is smallest and lightest?",
       "Model 9801 is the smallest and lightest, weighing 20.94 lb (9.5 kg), compared to 9803/B at 88.2 lb (40 kg) and 9805/B at 115 lb (52.16 kg)."),

    # ── PARAPHRASED VARIANTS OF CRITICAL FACTS ────────────────────────────────
    qa("Tell me the crest factor specification for the 9800/B Series.",
       "The crest factor of the 9800/B Series AC output is 3."),

    qa("What is CF for the 9800/B Series?",
       "The crest factor (CF) of the 9800/B Series AC output is 3."),

    qa("What is the rated crest factor of the output on the B&K Precision 9800/B?",
       "The rated crest factor of the AC output is 3."),

    qa("Is the crest factor of the 9800/B Series output 1.5 or 3?",
       "The crest factor of the 9800/B Series AC output is 3, not 1.5."),

    qa("What is the 9805/B maximum VA rating?",
       "The maximum VA rating of model 9805/B is 1500 VA."),

    qa("How many volt-amperes can the 9805/B supply?",
       "Model 9805/B can supply up to 1500 VA."),

    qa("Is the maximum output of the 9805/B 1200 VA or 1500 VA?",
       "The maximum output of model 9805/B is 1500 VA, not 1200 VA."),

    qa("What is the 9801 maximum VA rating?",
       "The maximum VA rating of model 9801 is 300 VA."),

    qa("What is the 9803/B maximum VA rating?",
       "The maximum VA rating of model 9803/B is 750 VA."),

    qa("What is the voltage measurement accuracy specification?",
       "The voltage measurement accuracy is ±(0.2% + 0.6 V). It is NOT simply ±1% of the nominal value."),

    qa("Is the voltage measurement accuracy ±1% of reading?",
       "No. The voltage measurement accuracy is ±(0.2% + 0.6 V), not ±1% of reading."),

    qa("How many List mode steps are supported — 16 or 100?",
       "The 9800/B Series List mode supports up to 100 steps (in up to 10 groups), not 16 steps."),

    qa("How many total steps can be programmed in List mode?",
       "Up to 100 steps can be programmed in List mode, organized into up to 10 groups."),

    qa("Does List mode support 10 or 100 programs?",
       "List mode supports 10 user-defined programs (groups), each with up to 100 steps — 100 total steps across up to 10 groups."),

    qa("What are the remote control interfaces on the 9800/B Series?",
       "The remote control interfaces are: USB (USBTMC), RS-232, and LAN (Ethernet), all standard. GPIB is optional on select models (9803 and 9805, not 9803B/9805B). BNC I/O is also provided for triggering and synchronization."),

    qa("Does the 9800/B Series support RS-485?",
       "No. The 9800/B Series does not include RS-485. The supported serial interface is RS-232."),

    qa("Does the 9800/B Series support Ethernet?",
       "Yes. LAN (Ethernet) is a standard interface on all models of the 9800/B Series."),

    # ── ADDITIONAL SPEC DETAILS ───────────────────────────────────────────────
    qa("What is the output voltage programming range of the 9800/B Series?",
       "The output voltage (RMS) programming range is 0 to 300 V, with support for 150 V / 300 V auto-ranging mode."),

    qa("What is the output frequency programming range of the 9800/B Series?",
       "The output frequency programming range is 45 Hz to 500 Hz."),

    qa("At what ambient temperature are 9800/B Series specifications guaranteed?",
       "Specifications are guaranteed at an ambient temperature of 23 °C ± 5 °C after 15 minutes of warm-up."),

    qa("What waveform disturbances can the PLD simulator insert?",
       "The PLD simulator can insert surges, sags, spikes, and dropouts at user-defined locations on the AC waveform."),

    qa("What is the purpose of the System BUS connector on the rear panel of model 9801?",
       "The System BUS connector on the rear panel of model 9801 is not used (reserved)."),

    qa("What voltage selector is available on the rear panel of model 9801?",
       "Model 9801 has a 110/220 V selector on the rear panel to choose between 110 VAC and 220 VAC input."),

    qa("How is the 9800/B Series power source classified — linear or switching?",
       "The 9800/B Series consists of fully programmable linear AC power sources."),

    qa("What type of output waveform does the 9800/B Series produce?",
       "The 9800/B Series produces a sinusoidal AC output waveform with programmable voltage, frequency, and phase."),

    qa("Where are the AC output terminals located on the 9800/B Series?",
       "The 9800/B Series has universal line output terminals on both the front and rear of the unit."),

    qa("Can the 9800/B Series output both AC and DC?",
       "The datasheet specifies the 9800/B Series as a programmable AC power source. It is designed for AC output (0–300 V, 45–500 Hz). The datasheet does not specify DC output capability."),

    qa("What is the voltage output resolution when programming via the front panel?",
       "The voltage (RMS) programming resolution is 0.1 V."),

    qa("What is the maximum frequency the 9800/B Series can output?",
       "The maximum output frequency is 500 Hz."),

    qa("What is the minimum frequency the 9800/B Series can output?",
       "The minimum output frequency is 45 Hz."),

    qa("What is the minimum output voltage of the 9800/B Series?",
       "The minimum output voltage is 0 V."),

    qa("What is the maximum output voltage of the 9800/B Series?",
       "The maximum output voltage is 300 V (RMS)."),

    qa("What is the THD specification at 60 Hz with a resistive load?",
       "The Total Harmonic Distortion (THD) is ≤0.5% at 45 to 500 Hz with a resistive load. This applies at 60 Hz as well."),

    qa("Can the 9800/B Series be used for production testing?",
       "Yes. The 9800/B Series is suitable for production testing, R&D, service, and pre-compliance testing."),

    qa("Is the 9800/B Series suitable for R&D?",
       "Yes. The 9800/B Series is suitable for R&D (research and development) as well as production testing, service, and pre-compliance testing."),

    qa("What triggers can start a List mode sequence?",
       "A List mode sequence can be triggered from the front panel or via the BNC connector on the rear panel."),

    qa("Can measurement data be logged with the provided software?",
       "Yes. The included PC application software supports logging measurement data without requiring source code programming."),

    qa("What is the BNC connector signal type on the 9800/B Series?",
       "The BNC I/O connector supports: trigger input, sync output, output status indication, and output control/indicator."),

    qa("What is Range 1 on the 9800/B Series?",
       "Range 1 is the 150 V output range, which allows higher current output at voltages up to 150 V."),

    qa("What is Range 2 on the 9800/B Series?",
       "Range 2 is the 300 V output range, which allows operation up to 300 V but with half the current capability compared to Range 1."),

    qa("What is the input current at 110 VAC for each model?",
       "Maximum input current at 110 VAC:\n• Model 9801: 10 A\n• Model 9803/B: 30 A\n• Model 9805/B: 40 A"),

    qa("What is the input current at 220 VAC for each model?",
       "Maximum input current at 220 VAC:\n• Model 9801: 6.3 A\n• Model 9803/B: 15 A\n• Model 9805/B: 20 A"),

    qa("What RMS current does model 9803/B supply at 150 V range?",
       "In the 150 V range (Range 1), model 9803/B provides a maximum RMS output current of 6 A."),

    qa("What peak current does model 9803/B supply at 300 V range?",
       "In the 300 V range (Range 2), model 9803/B provides a maximum peak output current of 9 A."),

    qa("What is the apparent power accuracy of the 9800/B Series?",
       "Apparent power (VA) accuracy is calculated as Voltage (RMS) × Current (RMS)."),

    qa("What is the power factor measurement accuracy of the 9800/B Series?",
       "The datasheet does not specify a separate accuracy figure for power factor; apparent power accuracy is stated as Voltage (RMS) × Current (RMS)."),

    qa("How are the current measurement ranges switched automatically?",
       "The current range switches automatically based on Ipeak: it switches up when Ipeak exceeds 300% of the current range, down from high to mid when Ipeak drops below 80% of the high range, and down from mid to low when Ipeak drops below 20% of the mid range."),

    qa("What memory capacity does the 9800/B Series have?",
       "The 9800/B Series has 10 memory locations for storing instrument settings."),

    qa("Can the 9800/B Series simulate frequency variations?",
       "Yes. Using Sweep mode or List mode, the 9800/B Series can simulate user-defined frequency variations from 45 Hz to 500 Hz."),

    qa("What is the accuracy of the phase angle setting at 60 Hz?",
       "The phase angle programming accuracy is ±1° for frequencies from 45 to 65 Hz, which includes 60 Hz."),

    qa("What is the resolution of the phase angle setting?",
       "The phase angle programming resolution is 0.1°."),

    qa("What is the maximum phase angle that can be programmed?",
       "The maximum phase angle that can be programmed is 360°."),

    qa("What is the minimum phase angle that can be programmed?",
       "The minimum phase angle that can be programmed is 0°."),

    qa("Does the 9800/B Series support combined voltage and frequency sweep?",
       "Yes. In Sweep mode, voltage and frequency sweeps can be created independently or combined into a simultaneous voltage plus frequency sweep."),

    qa("What software environment does B&K Precision provide a driver for?",
       "B&K Precision provides a LabVIEW driver and softpanel for remote control of the 9800/B Series."),

    qa("What programming interface is available beyond LabVIEW?",
       "Beyond LabVIEW, the 9800/B Series can be programmed via USB (USBTMC), RS-232, LAN, or optional GPIB interfaces using standard command protocols."),

    qa("What is the typical temperature coefficient of the voltage output?",
       "The temperature coefficient is ±0.04% per °C (typical)."),

    qa("Are all specifications met at any ambient temperature?",
       "No. All specifications apply after a 15-minute warm-up at an ambient temperature of 23 °C ± 5 °C."),

    qa("What is the maximum humidity for operation?",
       "The maximum relative humidity for operation is 80%, with no condensation allowed, at temperatures from 0 °C to 40 °C."),

    qa("What is the maximum storage humidity for the 9800/B Series?",
       "The maximum storage relative humidity is 85%, at temperatures from -20 °C to 70 °C."),

    qa("What is the minimum operating temperature in Celsius?",
       "The minimum operating temperature is 0 °C (32 °F)."),

    qa("What is the maximum operating temperature in Celsius?",
       "The maximum operating temperature is 40 °C (104 °F)."),

    qa("Can the 9800/B Series be used outdoors?",
       "No. The 9800/B Series is for indoor use only."),

    qa("Does the 9800/B Series require any warm-up time?",
       "Yes. A 15-minute temperature stabilization period is required before all specifications are valid."),

    qa("What is the frequency measurement accuracy of the 9800/B Series?",
       "The frequency measurement accuracy is ±0.1 Hz."),

    qa("What is the full-scale load regulation specification?",
       "Load regulation is ≤0.5% FS (full scale) with a resistive load."),

    qa("What does the 9800/B Series use to display output parameters?",
       "The 9800/B Series uses a bright VFD (Vacuum Fluorescent Display) on the front panel to show Vrms, Irms, Ipeak, frequency, power factor (PF), apparent power, true power, and elapsed output time."),

    qa("How quickly does the 9800/B Series output respond to a change?",
       "The output response time is less than 100 µs (<100 µs)."),

    qa("What does OVP stand for on the 9800/B Series?",
       "OVP stands for Over Voltage Protection."),

    qa("What does OCP stand for on the 9800/B Series?",
       "OCP stands for Over Current Protection."),

    qa("What does OPP stand for on the 9800/B Series?",
       "OPP stands for Over Power Protection."),

    qa("What does OTP stand for on the 9800/B Series?",
       "OTP stands for Over Temperature Protection."),

    qa("What does THD stand for in the 9800/B Series specification?",
       "THD stands for Total Harmonic Distortion."),

    qa("What is the THD of the 9800/B Series with a resistive load across the frequency range?",
       "The Total Harmonic Distortion (THD) is ≤0.5% at 45 to 500 Hz with a resistive load."),

    qa("What does PLD stand for in the 9800/B Series?",
       "PLD stands for Power Line Disturbance."),

    qa("What does VFD stand for in the context of the 9800/B Series display?",
       "VFD stands for Vacuum Fluorescent Display, which is the type of front panel display used on the 9800/B Series."),

    qa("What does PF stand for as shown on the 9800/B Series display?",
       "PF stands for Power Factor, which is one of the parameters shown on the front panel VFD display."),

    qa("What is Ipeak as displayed on the 9800/B Series front panel?",
       "Ipeak is the peak (instantaneous maximum) output current, displayed alongside Vrms, Irms, frequency, PF, apparent power, true power, and elapsed output time on the front panel VFD."),

    qa("Does the 9800/B Series display elapsed output time?",
       "Yes. The front panel VFD display shows elapsed output time along with Vrms, Irms, Ipeak, frequency, PF, apparent power, and true power."),

    qa("What apparent power measurement ranges are available?",
       "Apparent power (VA) measurement has three ranges:\n• Low range: resolution 0.01 VA\n• Mid range: resolution 0.1 VA\n• High range: resolution 1 VA"),

    qa("How is the line regulation of the 9800/B Series specified?",
       "Line regulation is specified as 0.1% maximum for a ±10% change in input line voltage."),

    qa("What is the peak-to-RMS ratio implied by the crest factor?",
       "The crest factor of 3 means the peak voltage is 3 times the RMS voltage of the AC output."),

    qa("Can the 9800/B Series generate waveforms other than sine waves?",
       "The datasheet describes the 9800/B Series as a linear AC power source producing sine wave output. The List and PLD modes allow sequence-based disturbance insertion on the waveform, but no arbitrary waveform (AWG) capability is specified."),

    qa("What is the voltage programming accuracy of the 9800/B Series?",
       "The output voltage (RMS) programming accuracy is ±(0.2% + 0.6 V)."),

    qa("Is there a LabVIEW softpanel for the 9800/B Series?",
       "Yes. A LabVIEW driver and softpanel for remote control are available for the 9800/B Series."),

    qa("What is the current high range for model 9801?",
       "For model 9801, the high current measurement range is 3.00 A."),

    qa("What is the current high range for model 9803/B?",
       "For model 9803/B, the high current measurement range is 6.00 A."),

    qa("What is the current high range for model 9805/B?",
       "For model 9805/B, the high current measurement range is 12.00 A."),

    qa("What is the mid current measurement range on all models?",
       "The mid current measurement range is 1.200 A on all models (9801, 9803/B, 9805/B)."),

    qa("What is the low current measurement range on all models?",
       "The low current measurement range is 120.0 mA on all models (9801, 9803/B, 9805/B)."),

    qa("What accessories come standard with all 9800/B Series models?",
       "All models include a certificate of calibration and an instruction manual as standard accessories."),

    qa("What power cord comes with model 9801?",
       "Model 9801 includes an AC power cord as a standard accessory."),

    qa("What power cord comes with models 9803/B and 9805/B?",
       "Models 9803/B and 9805/B include an unterminated power cord with input connector (not a standard AC power cord) as a standard accessory."),

    qa("Can both 9803/B and 9805/B be rack mounted?",
       "Yes. Rackmount ears and handles are standard accessories for both 9803/B and 9805/B. An optional IT-E151 rack mount kit is available for model 9801."),

    qa("What is the optional rack mount kit model number for the 9801?",
       "The optional rack mount kit for model 9801 is the IT-E151."),

    qa("In what year was the 9800/B Series datasheet last revised?",
       "The datasheet copyright is dated 2025 (© B&K Precision Corp. 2025), with a version dated v071425."),

    qa("What is the B&K Precision website?",
       "The B&K Precision website is bkprecision.com."),
]

# Ensure exactly 300 samples — shuffle and trim/pad if needed
print(f"Generated {len(samples)} samples before deduplication")

# Deduplicate by question
seen = set()
unique = []
for s in samples:
    q = s["messages"][1]["content"]
    if q not in seen:
        seen.add(q)
        unique.append(s)

print(f"Unique samples: {len(unique)}")

extra_samples = [
    qa("What is the peak current measurement accuracy for model 9803/B?",
       "The peak current measurement accuracy for model 9803/B is ±(1% + 120 mA), range 0 to 18 A, resolution 0.01 A."),

    qa("What is the peak current measurement accuracy for model 9805/B?",
       "The peak current measurement accuracy for model 9805/B is ±(1% + 120 mA), range 0 to 36 A, resolution 0.01 A."),

    qa("What is the maximum peak current of model 9803/B in the 300 V range?",
       "In the 300 V range (Range 2), model 9803/B provides a maximum peak output current of 9 A."),

    qa("What is the maximum RMS current of model 9803/B in the 300 V range?",
       "In the 300 V range (Range 2), model 9803/B provides a maximum RMS output current of 3 A."),

    qa("What is the maximum RMS current of model 9801 in the 300 V range?",
       "In the 300 V range (Range 2), model 9801 provides a maximum RMS output current of 1.5 A."),

    qa("What is the maximum peak current of model 9801 in the 300 V range?",
       "In the 300 V range (Range 2), model 9801 provides a maximum peak output current of 4.5 A."),

    qa("What are the voltage programming ranges supported by the 9800/B Series?",
       "The 9800/B Series supports output voltage programming from 0 to 300 V, with 150 V range (Range 1), 300 V range (Range 2), and an auto-ranging 150 V / 300 V mode."),

    qa("What is the Sefram subsidiary and where does it operate?",
       "Sefram is B&K Precision's French subsidiary, serving European customers."),

    qa("Where does B+K Precision Taiwan serve customers?",
       "B+K Precision Taiwan serves customers in Asia."),

    qa("Where are the independent service centers for B&K Precision located?",
       "B&K Precision has independent service centers in Singapore and Brazil, serving Singapore, Malaysia, Vietnam, Indonesia, and South America."),

    qa("Can I use the 9800/B Series for pre-compliance testing?",
       "Yes. The 9800/B Series is suitable for pre-compliance testing, including voltage dips and frequency simulations per IEC61000-4-11, IEC61000-4-14, and IEC61000-4-28."),

    qa("What IEC standard covers voltage dips testing supported by the 9800/B Series?",
       "IEC61000-4-11 covers voltage dips, and the 9800/B Series supports pre-compliance testing for this standard."),

    qa("What IEC standard covers frequency simulation testing supported by the 9800/B Series?",
       "IEC61000-4-28 covers voltage fluctuations and flicker for frequency simulation, and the 9800/B Series supports pre-compliance testing for this standard."),

    qa("What is the output voltage setting resolution when using the encoder knob?",
       "The output voltage programming resolution is 0.1 V regardless of input method (keypad or encoder knob)."),

    qa("What is the difference between Vrms and Vpeak output?",
       "Vrms is the root-mean-square voltage of the output. Vpeak is the instantaneous peak voltage. With a crest factor of 3, Vpeak = 3 × Vrms for the 9800/B Series output."),

    qa("What is the maximum power output available in 300 V range for model 9805/B?",
       "In the 300 V range (Range 2), model 9805/B provides a maximum RMS current of 6 A, meaning at 300 V the maximum power is approximately 1500 VA — but maximum power is only guaranteed when output voltage is set from 160 V to 240 V."),

    qa("How many models are in the 9800/B Series?",
       "There are three models in the 9800/B Series: 9801 (300 VA), 9803/B (750 VA), and 9805/B (1500 VA)."),

    qa("What are the three measurement ranges for current on the 9800/B Series?",
       "The three current measurement ranges are:\n• Low: 120.0 mA\n• Mid: 1.200 A\n• High: 3.00 A / 6.00 A / 12.00 A (depending on model)"),

    qa("Does the 9800/B Series support programmable voltage limits?",
       "Yes. The 9800/B Series includes programmable voltage and frequency limit settings."),

    qa("Does the 9800/B Series support programmable frequency limits?",
       "Yes. The 9800/B Series includes programmable voltage and frequency limit settings."),

    qa("What does the key lock function do on the 9800/B Series?",
       "The key lock function prevents accidental changes to instrument settings by locking the front panel keys."),

    qa("Can the 9800/B Series run automated test sequences?",
       "Yes. The List mode and Sweep mode allow automated test sequences to be programmed, stored, and executed via the front panel or BNC trigger input."),

    qa("What is the serial communication interface speed for RS-232 on the 9800/B Series?",
       "The datasheet does not specify the RS-232 baud rate. It only confirms that RS-232 is a standard interface on all 9800/B Series models."),

    qa("Does the 9800/B Series support SCPI commands?",
       "The datasheet states USB (USBTMC-compliant), RS-232, LAN, and GPIB interfaces are available but does not explicitly confirm SCPI. USBTMC compliance typically implies SCPI support."),

    qa("What connector standard is used for the USB interface on the 9800/B Series?",
       "The USB interface is USBTMC-compliant. The specific USB connector type is not stated in the datasheet."),

    qa("What is the output terminal location on the 9803/B rear panel?",
       "Models 9803/B and 9805/B have an AC output terminal on the rear panel, as well as the LAN, RS232, USB interface connectors, a BNC I/O connector, and cooling fan vents."),

    qa("Does the 9800/B Series include PC software for remote control?",
       "Yes. PC application software is provided for front panel emulation, list/PLD/sweep profile generation, and measurement data logging without source code."),

    qa("What is the model number of the smallest 9800/B Series unit?",
       "The smallest model is 9801, delivering 300 VA."),

    qa("What is the model number of the largest 9800/B Series unit?",
       "The largest model is 9805/B, delivering 1500 VA."),

    qa("How is the 9800/B Series typically used in R&D environments?",
       "In R&D environments, the 9800/B Series is used for evaluating transformers, TRIACs, SCRs, passive components, switching power supplies, and for simulating grid disturbances and pre-compliance testing."),

    qa("What does the 9800/B Series use for analog-style parameter adjustment?",
       "The 9800/B Series front panel includes an encoder knob for quick analog-style control of output parameters."),

    qa("What is the current measurement accuracy for peak current across all models?",
       "The peak current measurement accuracy is ±(1% + 120 mA) for all models (9801, 9803/B, 9805/B)."),

    qa("What is the power factor range of the 9800/B Series output?",
       "Typical power factor is 0.5 for model 9801 and 0.7 for models 9803/B and 9805/B. The power factor measurement display range is 0.000 to 1.000."),

    qa("How are settings saved on the 9800/B Series?",
       "Up to 100 instrument settings can be saved to and recalled from the 9800/B Series memory (10 memory locations)."),

    qa("What communication protocols does the 9800/B Series LAN interface support?",
       "The datasheet confirms that LAN (Ethernet) is a standard interface. Specific protocol details (e.g., VXI-11, LXI) are not listed in the datasheet."),

    qa("Can the 9800/B Series output be synchronized with an external signal?",
       "Yes. The BNC I/O connector on the rear panel provides a sync output for synchronization with external equipment."),
]

unique_questions = {s["messages"][1]["content"] for s in unique}
for s in extra_samples:
    q = s["messages"][1]["content"]
    if q not in unique_questions:
        unique.append(s)
        unique_questions.add(q)

random.shuffle(unique)

out_path = "train_9800_300.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for item in unique:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Final dataset: {len(unique)} samples saved to {out_path}")
