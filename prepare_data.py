"""
prepare_data.py
───────────────────────────────────────────────────────────────
Extracts text from 9800_Series_datasheet.pdf and generates a
domain-specific Q&A dataset (train_9800.jsonl) for fine-tuning
Qwen2.5-0.5B-Instruct.

Usage:
    python prepare_data.py

Output:
    train_9800.jsonl   — JSONL file with chat-format messages
"""

import json
import re
import pdfplumber

PDF_PATH  = "9800_Series_datasheet.pdf"
OUT_PATH  = "train_9800.jsonl"
SYSTEM    = (
    "You are an expert technical assistant for B&K Precision 9800/B Series "
    "Programmable AC Power Sources. Answer questions accurately and concisely "
    "based on the official datasheet specifications."
)

# ─────────────────────────────────────────────────────────────
# STEP 1 — Extract raw text from PDF
# ─────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    """Return concatenated plain text from all pages."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
    return "\n\n".join(pages)


# ─────────────────────────────────────────────────────────────
# STEP 2 — Hand-crafted Q&A pairs grounded in the datasheet
#          (These are derived directly from the extracted text)
# ─────────────────────────────────────────────────────────────

QA_PAIRS = [
    # ── General / Overview ───────────────────────────────────
    {
        "q": "What is the 9800/B Series?",
        "a": (
            "The 9800/B Series is a line of programmable AC power sources and "
            "measurement tools manufactured by B&K Precision. These fully "
            "programmable linear AC sources deliver up to 1500 VA through "
            "universal line output terminals on the front and rear of the unit."
        ),
    },
    {
        "q": "What models are included in the 9800/B Series?",
        "a": (
            "The 9800/B Series includes three models:\n"
            "• Model 9801 — up to 300 VA\n"
            "• Model 9803/B — up to 750 VA\n"
            "• Model 9805/B — up to 1500 VA"
        ),
    },
    {
        "q": "What is the output voltage range of the 9800/B Series?",
        "a": (
            "The 9800/B Series provides an output voltage range of 0 to 300 V "
            "(AC). The units support a 150 V / 300 V autoranging mode or "
            "300 V range operation for a continuous voltage sweep."
        ),
    },
    {
        "q": "What is the output frequency range of the 9800/B Series?",
        "a": (
            "The output frequency is adjustable from 45 Hz to 500 Hz."
        ),
    },
    {
        "q": "What AC input does the 9800/B Series require?",
        "a": (
            "The AC input range is single-phase 110/220 VAC ±10%, "
            "at a line frequency of 47 to 63 Hz."
        ),
    },

    # ── Model-specific specs ──────────────────────────────────
    {
        "q": "What is the maximum power output of model 9801?",
        "a": "Model 9801 delivers a maximum power output of 300 VA.",
    },
    {
        "q": "What is the maximum power output of model 9803/B?",
        "a": "Model 9803/B delivers a maximum power output of 750 VA.",
    },
    {
        "q": "What is the maximum power output of model 9805/B?",
        "a": "Model 9805/B delivers a maximum power output of 1500 VA.",
    },
    {
        "q": "What is the maximum RMS current of model 9805/B in the 150 V range?",
        "a": (
            "In the 150 V range (Range 1), model 9805/B provides a maximum "
            "RMS current of 12 A."
        ),
    },
    {
        "q": "What is the maximum peak current of model 9805/B in the 150 V range?",
        "a": (
            "In the 150 V range (Range 1), model 9805/B provides a maximum "
            "peak current of 36 A."
        ),
    },
    {
        "q": "What is the maximum RMS current for model 9801 at 110 VAC input?",
        "a": (
            "At 110 VAC input, model 9801 draws a maximum input current of 10 A."
        ),
    },
    {
        "q": "What is the crest factor of the 9800/B Series output?",
        "a": "The crest factor of the AC output is 3.",
    },
    {
        "q": "What is the power factor of model 9801?",
        "a": "Model 9801 has a typical power factor of 0.5.",
    },
    {
        "q": "What is the power factor of models 9803/B and 9805/B?",
        "a": "Models 9803/B and 9805/B have a typical power factor of 0.7.",
    },

    # ── Voltage measurement specs ─────────────────────────────
    {
        "q": "What is the voltage measurement resolution of the 9800/B Series?",
        "a": "The voltage measurement resolution is 0.1 V.",
    },
    {
        "q": "What is the voltage measurement accuracy of the 9800/B Series?",
        "a": "The voltage measurement accuracy is ±(0.2% + 0.6 V).",
    },

    # ── Current measurement specs ─────────────────────────────
    {
        "q": "What are the current measurement ranges available on the 9800/B Series?",
        "a": (
            "There are three current measurement ranges:\n"
            "• Low range: 120.0 mA (resolution: 0.1 mA)\n"
            "• Mid range: 1.200 A (resolution: 1 mA)\n"
            "• High range: 3.00 A / 6.00 A / 12.00 A depending on model "
            "  (resolution: 10 mA)"
        ),
    },
    {
        "q": "What is the peak current measurement range of the 9800/B Series?",
        "a": (
            "The peak current measurement range is:\n"
            "• 9801: 0 to 9 A (resolution 0.01 A, accuracy ±(1% + 120 mA))\n"
            "• 9803/B: 0 to 18 A\n"
            "• 9805/B: 0 to 36 A"
        ),
    },

    # ── True power measurement ────────────────────────────────
    {
        "q": "What is the true power measurement resolution of the 9800/B Series?",
        "a": (
            "True power (watts) resolution:\n"
            "• Low range: 0.01 W\n"
            "• Mid range: 0.1 W\n"
            "• High range: 1 W"
        ),
    },
    {
        "q": "What is the true power measurement accuracy at low range?",
        "a": (
            "At low range, true power accuracy is ±(0.2% + 0.05 W), valid for "
            "frequencies from 47 to 65 Hz."
        ),
    },

    # ── Frequency specs ───────────────────────────────────────
    {
        "q": "What is the frequency resolution of the 9800/B Series?",
        "a": (
            "Frequency resolution is ±0.1 Hz for the range 45 to 99.9 Hz, "
            "and ±1 Hz for the range 100 to 500 Hz."
        ),
    },

    # ── Features ──────────────────────────────────────────────
    {
        "q": "What is List mode on the 9800/B Series?",
        "a": (
            "List mode allows generation of complex waveform sequences with "
            "varying times, amplitudes, and frequencies. Up to 100 steps in "
            "10 groups can be saved and executed. This is useful for simulating "
            "grid faults and disturbances such as surges, sags, and frequency "
            "variations. The programmed list can be triggered from the front "
            "panel or via a BNC connector on the rear."
        ),
    },
    {
        "q": "What is the PLD simulator in the 9800/B Series?",
        "a": (
            "The Power Line Disturbance (PLD) simulator is an extended feature "
            "of List mode that gives the user finer control over disturbance "
            "insertion into the AC waveform. It is useful for evaluating a "
            "product's immunity performance by producing common waveform "
            "disturbances such as surges, sags, and spikes."
        ),
    },
    {
        "q": "What is Sweep mode on the 9800/B Series?",
        "a": (
            "Sweep mode enables user-defined voltage and frequency sweeps, "
            "which can be created independently or combined. It is ideal for "
            "testing the efficiency of switching power supplies or capturing the "
            "maximum operating power requirements of a device under test. "
            "Up to 10 sweep profiles can be stored and recalled."
        ),
    },
    {
        "q": "What is the Dimmer simulation feature of the 9800/B Series?",
        "a": (
            "The Dimmer simulation feature controls the phase cut-off of the AC "
            "sine wave's leading or trailing edge, varying the RMS voltage "
            "supplied to the load under test. The phase cut-off is adjustable "
            "between 0° and 180° for both leading-edge and trailing-edge "
            "dimming. It is useful for motor control and lighting test "
            "applications."
        ),
    },

    # ── Display ───────────────────────────────────────────────
    {
        "q": "What does the front panel display show on the 9800/B Series?",
        "a": (
            "The bright VFD (Vacuum Fluorescent Display) shows: Vrms, Irms, "
            "Ipeak, frequency, power factor (PF), apparent power, true power, "
            "and elapsed output time."
        ),
    },
    {
        "q": "What front panel controls are available on the 9800/B Series?",
        "a": (
            "The front panel features:\n"
            "• A numeric keypad for direct entry of AC output waveform parameters\n"
            "• Function keys\n"
            "• Cursor keys\n"
            "• An encoder knob for quick analog-style control\n"
            "• An AC output terminal"
        ),
    },

    # ── Interfaces ────────────────────────────────────────────
    {
        "q": "What remote interfaces does the 9800/B Series support?",
        "a": (
            "The 9800/B Series supports the following remote interfaces:\n"
            "• USB\n"
            "• RS-232\n"
            "• LAN (Ethernet)\n"
            "• GPIB (optional, on select models)\n"
            "• BNC I/O for triggering, output status/control, and synchronization"
        ),
    },
    {
        "q": "Is GPIB available on all models of the 9800/B Series?",
        "a": (
            "No. GPIB is an optional interface available only on select models "
            "of the 9800/B Series. Refer to the ordering information for details."
        ),
    },
    {
        "q": "What is the BNC connector used for on the 9800/B Series?",
        "a": (
            "The BNC I/O connector on the rear panel is used for triggering, "
            "output status/control, and synchronization. The programmed List "
            "mode can also be triggered via this BNC connector."
        ),
    },

    # ── Rear panel ────────────────────────────────────────────
    {
        "q": "What connectors are on the rear panel of model 9801?",
        "a": (
            "The rear panel of model 9801 includes:\n"
            "• AC line input\n"
            "• 110/220 V selector\n"
            "• BNC I/O trigger connector\n"
            "• AC output terminal\n"
            "• System BUS\n"
            "• RS-232 interface\n"
            "• LAN interface\n"
            "• USB interface\n"
            "• Optional GPIB interface"
        ),
    },
    {
        "q": "What cooling mechanism is used in the 9803/B and 9805/B models?",
        "a": (
            "Models 9803/B and 9805/B use cooling fans for thermal management."
        ),
    },

    # ── Company ───────────────────────────────────────────────
    {
        "q": "Who manufactures the 9800/B Series AC power sources?",
        "a": (
            "The 9800/B Series is manufactured by B&K Precision Corporation, "
            "headquartered in Yorba Linda, California, USA. B&K Precision has "
            "provided test and measurement instruments worldwide for more than "
            "70 years and is an ISO 9001:2015 registered company."
        ),
    },
    {
        "q": "What quality certification does B&K Precision hold?",
        "a": (
            "B&K Precision Corporation is ISO 9001:2015 registered, employing "
            "traceable quality management practices for all processes including "
            "product development, service, and calibration."
        ),
    },

    # ── Comparison questions ──────────────────────────────────
    {
        "q": "How does model 9805/B differ from model 9801?",
        "a": (
            "Key differences:\n"
            "• Power: 9805/B = 1500 VA vs 9801 = 300 VA\n"
            "• Max RMS current (150 V range): 9805/B = 12 A vs 9801 = 3 A\n"
            "• Max peak current (150 V range): 9805/B = 36 A vs 9801 = 9 A\n"
            "• Input current at 110 VAC: 9805/B = 40 A vs 9801 = 10 A\n"
            "• Power factor: 9805/B = 0.7 (typical) vs 9801 = 0.5 (typical)\n"
            "• 9805/B uses cooling fans; 9801 does not."
        ),
    },
    {
        "q": "What is the difference between the 150 V range and 300 V range on the 9800/B Series?",
        "a": (
            "In the 150 V range (Range 1), the unit can deliver higher current "
            "at lower voltages. In the 300 V range (Range 2), the maximum "
            "current is halved but voltage capability doubles. For example, on "
            "the 9805/B: 150 V range = 12 A RMS / 36 A peak, "
            "300 V range = 6 A RMS / 18 A peak."
        ),
    },

    # ── Application questions ─────────────────────────────────
    {
        "q": "What applications is the 9800/B Series suitable for?",
        "a": (
            "The 9800/B Series is suitable for a wide range of applications "
            "including:\n"
            "• Testing switching power supply efficiency (sweep mode)\n"
            "• Simulating grid faults and disturbances (list mode / PLD)\n"
            "• Motor control and lighting testing (dimmer simulation)\n"
            "• Product immunity testing against surges, sags, and spikes\n"
            "• General-purpose programmable AC power sourcing and measurement"
        ),
    },
    {
        "q": "Can the 9800/B Series simulate power line surges and sags?",
        "a": (
            "Yes. Using the List mode and the Power Line Disturbance (PLD) "
            "simulator, the 9800/B Series can generate sequences of waveforms "
            "that simulate power line disturbances including surges, sags, "
            "frequency variations, and voltage spikes."
        ),
    },
    {
        "q": "How many sweep profiles can be stored in the 9800/B Series?",
        "a": "Up to 10 sweep profiles can be stored and recalled.",
    },
    {
        "q": "How many list mode steps and groups does the 9800/B Series support?",
        "a": (
            "The 9800/B Series supports up to 100 steps organized in up to "
            "10 groups for List mode operation."
        ),
    },
    {
        "q": "What dimmer edge modes does the 9800/B Series support?",
        "a": (
            "The 9800/B Series supports both leading-edge and trailing-edge "
            "dimmer simulation. The phase cut-off angle is adjustable from "
            "0° to 180° for both modes."
        ),
    },
]


# ─────────────────────────────────────────────────────────────
# STEP 3 — Write JSONL
# ─────────────────────────────────────────────────────────────

def write_jsonl(qa_pairs: list, out_path: str, system: str) -> int:
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in qa_pairs:
            record = {
                "messages": [
                    {"role": "system",    "content": system},
                    {"role": "user",      "content": item["q"]},
                    {"role": "assistant", "content": item["a"]},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print(f"📄 Extracting text from {PDF_PATH}...")
    raw_text = extract_pdf_text(PDF_PATH)
    pages = raw_text.count("\n\n") + 1
    words = len(raw_text.split())
    print(f"   → Extracted ~{words} words across {pages} sections")

    print(f"\n✍️  Writing Q&A dataset to {OUT_PATH}...")
    n = write_jsonl(QA_PAIRS, OUT_PATH, SYSTEM)
    print(f"   → Wrote {n} training examples")

    print(f"\n✅ Dataset ready: {OUT_PATH}")
    print(f"   Total examples : {n}")
    print(f"   System prompt  : {SYSTEM[:60]}...")
    print("\nNext step: run   python train.py")


if __name__ == "__main__":
    main()
