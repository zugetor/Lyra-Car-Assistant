"""
train_phrases.py  --  phrase bank for gen_train_data.py

Many ways to say the same command, per control and payload, in Thai and English.
{o} is replaced by every object word of the control (e.g. ฝาท้าย / ประตูหลัง / ท้ายรถ),
{n} by a number. gen_train_data.py adds politeness prefixes/suffixes on top.

To support a new way of saying something, add it here and re-run gen_train_data.py.
Keys are the real control keys (as in eval_commands.tsv), not the labels the model sees.
"""

# --------------------------------------------------------------------------- wrappers
TH_PREFIX = ["", "ช่วย", "ขอ", "รบกวน", "ช่วย", "ขอ"]          # only in front of a verb
TH_SUFFIX = ["", "หน่อย", "ด้วย", "ให้หน่อย", "ที", "ครับ", "ค่ะ", "นะ", "หน่อยครับ", "หน่อยค่ะ", "ให้หน่อยครับ", "เลย"]
EN_PREFIX = ["", "please ", "can you ", "could you ", "hey, ", "", ""]
EN_SUFFIX = ["", " please", " now", " for me", ""]

# --------------------------------------------------------------------------- OPEN / CLOSE / STOP
MOTION = {
    "windows_all": {
        "th_obj": ["กระจก", "กระจกทุกบาน", "กระจกทั้งหมด", "หน้าต่าง", "หน้าต่างทุกบาน", "กระจกรถ", "กระจกหน้าต่าง",
                   "หน้าต่างรถ", "กระจกทั้งคัน"],
        "en_obj": ["the windows", "all windows", "all the windows", "every window", "the car windows", "windows",
                   "all four windows"],
        "th": {"OPEN": ["เปิด{o}", "ลด{o}ลง", "ลด{o}", "เปิด{o}ลง"],
               "CLOSE": ["ปิด{o}", "ขึ้น{o}", "ยก{o}ขึ้น", "ปิด{o}ให้สนิท"],
               "STOP": ["หยุด{o}", "หยุดเลื่อน{o}", "{o}หยุดก่อน", "พอแล้ว หยุด{o}"]},
        "en": {"OPEN": ["open {o}", "roll down {o}", "lower {o}", "put {o} down"],
               "CLOSE": ["close {o}", "shut {o}", "roll up {o}", "raise {o}", "put {o} up"],
               "STOP": ["stop {o}", "stop moving {o}", "hold {o} there", "freeze {o}"]},
    },
    "tailgate": {
        "th_obj": ["ฝาท้าย", "ประตูหลัง", "ท้าย", "ท้ายรถ", "ประตูท้าย", "ฝากระโปรงท้าย", "ฝาหลัง", "ประตูท้ายรถ",
                   "ที่เก็บของท้ายรถ", "กระโปรงหลัง"],
        "en_obj": ["the tailgate", "the trunk", "the boot", "the rear hatch", "the back door", "the liftgate",
                   "the hatch", "the trunk lid"],
        "th": {"OPEN": ["เปิด{o}", "ยก{o}ขึ้น", "เปิด{o}ขึ้น"],
               "CLOSE": ["ปิด{o}", "ปิด{o}ลง", "ปิด{o}ให้สนิท"],
               "STOP": ["หยุด{o}", "{o}หยุดก่อน", "หยุดเลื่อน{o}"]},
        "en": {"OPEN": ["open {o}", "pop {o}", "lift {o}", "open up {o}"],
               "CLOSE": ["close {o}", "shut {o}", "close {o} down"],
               "STOP": ["stop {o}", "stop moving {o}", "hold {o}"]},
    },
    "sunroof": {
        "th_obj": ["ซันรูฟ", "หลังคา", "หลังคากระจก", "หลังคาซันรูฟ", "กระจกหลังคา", "มูนรูฟ", "หลังคาแก้ว"],
        "en_obj": ["the sunroof", "the moonroof", "the roof", "the glass roof", "the panoramic roof", "sunroof"],
        "th": {"OPEN": ["เปิด{o}", "เลื่อน{o}เปิด", "แง้ม{o}"],
               "CLOSE": ["ปิด{o}", "เลื่อน{o}ปิด"],
               "STOP": ["หยุด{o}", "{o}หยุดก่อน", "หยุดเลื่อน{o}"]},
        "en": {"OPEN": ["open {o}", "slide {o} open", "tilt {o} open"],
               "CLOSE": ["close {o}", "shut {o}", "slide {o} closed"],
               "STOP": ["stop {o}", "stop moving {o}", "hold {o} there"]},
    },
    "sunshade": {
        "th_obj": ["ม่านหลังคา", "ม่านบังแดด", "ม่านซันรูฟ", "ม่านบังแดดหลังคา", "ที่บังแดดหลังคา", "ม่าน",
                   "ม่านกันแดด", "ม่านใต้ซันรูฟ"],
        "en_obj": ["the sunshade", "the sun shade", "the roof shade", "the roof blind", "the sunroof shade",
                   "the shade", "the sun blind"],
        "th": {"OPEN": ["เปิด{o}", "เลื่อน{o}ออก", "เก็บ{o}"],
               "CLOSE": ["ปิด{o}", "ดึง{o}ปิด", "กาง{o}"],
               "STOP": ["หยุด{o}", "{o}หยุดก่อน", "หยุดเลื่อน{o}"]},
        "en": {"OPEN": ["open {o}", "pull back {o}", "retract {o}"],
               "CLOSE": ["close {o}", "draw {o}", "pull {o} closed"],
               "STOP": ["stop {o}", "stop moving {o}", "hold {o}"]},
    },
}

# --------------------------------------------------------------------------- climate
AIRCON = {  # climate/mode: auto / off
    "th": {"auto": ["เปิดแอร์", "เปิดเครื่องปรับอากาศ", "เปิดระบบปรับอากาศ", "แอร์ออโต้", "ตั้งแอร์เป็นออโต้",
                    "เปิดแอร์อัตโนมัติ", "เปิดแอร์แบบออโต้", "ร้อนจัง เปิดแอร์", "เปิดแอร์ที", "ตั้งแอร์อัตโนมัติ",
                    "เปิดแอร์รถ", "สตาร์ทแอร์"],
           "off": ["ปิดแอร์", "ปิดเครื่องปรับอากาศ", "ปิดระบบปรับอากาศ", "ไม่เอาแอร์แล้ว", "ปิดแอร์ก่อน",
                   "หยุดแอร์", "ปิดแอร์รถ", "เลิกเปิดแอร์"]},
    "en": {"auto": ["turn on the AC", "turn on the air conditioning", "AC on", "start the air con", "switch on the AC",
                    "set the AC to auto", "auto climate", "turn the aircon on", "put the AC on auto",
                    "turn on the climate", "climate on", "switch the air conditioner on"],
           "off": ["turn off the AC", "AC off", "switch off the air conditioning", "kill the AC", "stop the air con",
                   "shut off the climate", "turn the aircon off", "climate off", "switch the AC off"]},
}

TEMP_NUMBER = {  # climate/temperature with a number (the number is read by rule)
    "th": ["ตั้งอุณหภูมิ {n} องศา", "แอร์ {n} องศา", "ปรับแอร์เป็น {n}", "ขอ {n} องศา", "อุณหภูมิ {n}",
           "ตั้งแอร์ {n} องศา", "เอา {n} องศา", "ปรับอุณหภูมิเป็น {n} องศา", "เปิดแอร์ {n} องศา",
           "ตั้งแอร์ที่ {n}", "{n} องศา", "ปรับเป็น {n} องศา", "ปรับอุณหภูมิ {n}"],
    "en": ["set the temperature to {n}", "{n} degrees", "make it {n} degrees", "set the AC to {n}",
           "temperature {n}", "change the temperature to {n} degrees", "I want {n} degrees",
           "turn on the AC to {n} degrees", "set it to {n} degrees", "temperature to {n} degrees Celsius",
           "{n} degrees Celsius", "adjust the temperature to {n}"],
    "numbers": list(range(17, 34)),
}

TEMP_DIR = {  # climate/temperature without a number: warmer / cooler
    "th": {"warmer": ["ฉันหนาว", "หนาว", "หนาวจัง", "หนาวมาก", "เย็นไป", "แอร์เย็นไป", "เพิ่มอุณหภูมิ",
                      "ปรับให้อุ่นขึ้น", "อุ่นขึ้นอีก", "ลดความเย็น", "ลดแอร์ลง หนาวแล้ว", "ขึ้นอุณหภูมิ",
                      "หนาวสั่นเลย", "แอร์เย็นเกินไป", "ปรับอุณหภูมิขึ้น", "ขออุ่นกว่านี้"],
           "cooler": ["ร้อน", "ร้อนจัง", "ฉันร้อน", "ร้อนมาก", "ลดอุณหภูมิ", "ลดอุณหภูมิลง", "เย็นกว่านี้",
                      "ขอเย็นๆ", "ทำให้เย็นลง", "เพิ่มความเย็น", "แอร์ไม่เย็นเลย", "ร้อนเหงื่อแตก",
                      "ปรับอุณหภูมิลง", "เย็นอีกนิด", "ร้อนอบอ้าว"]},
    "en": {"warmer": ["I'm cold", "it's cold", "it's freezing in here", "make it warmer", "warmer", "turn up the heat",
                      "raise the temperature", "a bit warmer", "turn the temperature up", "it's too cold",
                      "I'm freezing", "warm it up", "too much AC", "increase the temperature"],
           "cooler": ["I'm hot", "it's too hot", "make it cooler", "cooler", "lower the temperature",
                      "turn the temperature down", "cool it down", "it's boiling in here", "a bit cooler",
                      "it's hot in here", "I'm sweating", "decrease the temperature", "colder"]},
}

FAN_NUMBER = {  # climate/fan_mode with a number
    "th": ["พัดลมระดับ {n}", "ปรับลมเป็น {n}", "ความแรงลม {n}", "ลมแอร์ระดับ {n}", "ตั้งพัดลม {n}",
           "ปรับพัดลมเป็นระดับ {n}", "ลมระดับ {n}", "เปิดพัดลมเบอร์ {n}", "แรงลมเบอร์ {n}"],
    "en": ["fan speed {n}", "set the fan to {n}", "fan level {n}", "blower to {n}", "set the fan speed to level {n}",
           "fan to level {n}", "change the fan to {n}", "put the fan on {n}"],
    "numbers": list(range(1, 8)),
}

FAN_DIR = {  # climate/fan_mode without a number: up / down
    "th": {"up": ["เพิ่มลม", "แรงลมขึ้น", "เร่งพัดลม", "ลมแรงขึ้น", "เปิดลมแรงๆ", "ลมเบาไป", "เพิ่มพัดลม",
                  "เร่งลมแอร์", "ลมไม่ค่อยมา", "เพิ่มแรงลม"],
           "down": ["ลดลม", "ลมเบาลง", "ลดพัดลม", "ลมแรงไป", "เบาลม", "ลดแรงลม", "ลมแอร์แรงเกินไป", "ลดลมแอร์"]},
    "en": {"up": ["more air", "fan up", "turn up the fan", "increase the fan", "stronger fan", "more fan",
                  "blow more air", "speed up the fan", "fan higher"],
           "down": ["less air", "fan down", "turn down the fan", "decrease the fan", "weaker fan", "too much air",
                    "slow the fan down", "fan lower", "less fan"]},
}

# --------------------------------------------------------------------------- seats
SEAT_VENT = {  # seat_vent_driver / seat_vent_passenger: off / low / medium / high
    "seats": {"seat_vent_driver": {"th": ["เบาะคนขับ", "เบาะฝั่งคนขับ", "เบาะผม", "เบาะฉัน", "ที่นั่งคนขับ"],
                                   "en": ["driver seat", "driver's seat", "driver side seat"]},
              "seat_vent_passenger": {"th": ["เบาะผู้โดยสาร", "เบาะฝั่งผู้โดยสาร", "เบาะข้างคนขับ", "ที่นั่งผู้โดยสาร",
                                             "เบาะซ้าย"],
                                      "en": ["passenger seat", "front passenger seat", "passenger side seat",
                                             "co-driver seat"]}},
    "vent": {"th": ["ระบายอากาศ", "พัดลม", "ลมเย็น", "ช่องลม", "ระบบระบายอากาศ"],   # seats already say เบาะ
             "en": ["ventilation", "cooling", "seat fan", "ventilated seat", "seat cooling"]},
    "levels": {"th": {"low": ["ระดับต่ำ", "เบาๆ", "ระดับ 1", "ต่ำ"], "medium": ["ระดับกลาง", "ปานกลาง", "ระดับ 2", "กลาง"],
                      "high": ["ระดับสูง", "แรงสุด", "ระดับ 3", "สูงสุด", "แรงๆ"]},
               "en": {"low": ["low", "on low", "level one", "to low"], "medium": ["medium", "on medium", "level two",
                                                                                 "to medium"],
                      "high": ["high", "on high", "level three", "to max", "to high"]}},
    "th": {"on": ["เปิด{v}{s}{l}", "เปิด{v}{s} {l}", "{v}{s}{l}", "ตั้ง{v}{s}{l}", "ปรับ{v}{s}เป็น{l}"],
           "off": ["ปิด{v}{s}", "ปิด{v}ของ{s}", "ไม่เอา{v}{s}แล้ว"]},
    "en": {"on": ["turn on the {s} {v} {l}", "set the {s} {v} {l}", "{s} {v} {l}", "switch the {s} {v} {l}",
                  "put the {s} {v} {l}"],
           "off": ["turn off the {s} {v}", "switch off the {s} {v}", "{s} {v} off", "stop the {s} {v}"]},
}

# --------------------------------------------------------------------------- on / off switches
SWITCH = {
    "drl": {"th_obj": ["ไฟเดย์ไลท์", "ไฟวิ่งกลางวัน", "ไฟกลางวัน", "เดย์ไลท์", "ไฟ DRL", "ไฟเดย์ไลท์รถ"],
            "en_obj": ["the daytime running lights", "the DRL", "the DRLs", "daytime lights", "the day lights",
                       "daytime running lights"]},
    "wireless_charging": {"th_obj": ["ที่ชาร์จไร้สาย", "ชาร์จไร้สาย", "แท่นชาร์จไร้สาย", "ที่ชาร์จมือถือไร้สาย",
                                     "ชาร์จมือถือแบบไร้สาย", "แท่นชาร์จมือถือ", "ที่ชาร์จโทรศัพท์"],
                          "en_obj": ["wireless charging", "the wireless charger", "the phone charger",
                                     "the charging pad", "wireless phone charging", "the phone charging pad"]},
}
SWITCH_VERBS = {"th": {"on": ["เปิด{o}", "เปิดใช้{o}", "ใช้{o}"], "off": ["ปิด{o}", "ปิดใช้{o}", "ไม่ใช้{o}", "หยุด{o}"]},
                "en": {"on": ["turn on {o}", "switch on {o}", "enable {o}", "{o} on", "activate {o}"],
                       "off": ["turn off {o}", "switch off {o}", "disable {o}", "{o} off", "deactivate {o}"]}}

# --------------------------------------------------------------------------- media
MEDIA = {
    "th": {"play": ["เปิดเพลง", "เล่นเพลง", "เล่นต่อ", "เล่นเพลงต่อ", "ขอฟังเพลง", "เปิดเพลงฟัง", "อยากฟังเพลง",
                    "เปิดเพลงต่อ", "เล่น", "เปิดเพลงที่ฟังค้างไว้"],
           "pause": ["หยุดเพลง", "พักเพลง", "หยุดเล่น", "ปิดเพลง", "หยุดเพลงก่อน", "พอก่อน หยุดเพลง", "หยุดชั่วคราว",
                     "พักเพลงไว้ก่อน"],
           "next": ["เพลงถัดไป", "ข้ามเพลง", "เปลี่ยนเพลง", "เพลงต่อไป", "ข้ามเพลงนี้", "ไปเพลงต่อไป", "ไม่ชอบเพลงนี้ ข้าม",
                    "เพลงหน้า"],
           "previous": ["เพลงก่อนหน้า", "ย้อนเพลง", "เพลงที่แล้ว", "ย้อนกลับเพลง", "กลับไปเพลงก่อน", "ย้อนไปเพลงเมื่อกี้",
                        "เอาเพลงเมื่อกี้"],
           "volume_up": ["เพิ่มเสียง", "เสียงดังขึ้น", "เร่งเสียง", "ดังขึ้นหน่อย", "เพิ่มเสียงเพลง", "เสียงเบาไป",
                         "เปิดเสียงดังๆ", "เร่งเสียงเพลง", "ไม่ค่อยได้ยิน เพิ่มเสียง"],
           "volume_down": ["ลดเสียง", "เบาเสียง", "เสียงเบาลง", "ลดเสียงลง", "เบาลงหน่อย", "เสียงดังไป", "ลดเสียงเพลง",
                           "เบาเพลงลง"],
           "mute": ["ปิดเสียง", "ปิดเสียงเพลง", "เงียบ", "ปิดเสียงหมด", "ขอเงียบๆ", "ปิดเสียงทั้งหมด"]},
    "en": {"play": ["play music", "play some music", "resume", "resume the music", "play", "start the music",
                    "put some music on", "continue the song", "play my playlist", "unpause"],
           "pause": ["pause", "pause the music", "stop the music", "stop playing", "hold the music", "pause the song",
                     "stop the song"],
           "next": ["next song", "next track", "skip", "skip this track", "play the next one", "skip this song",
                    "next", "change the song"],
           "previous": ["previous song", "previous track", "go back a song", "last song", "play the previous one",
                        "back one song", "replay the last song"],
           "volume_up": ["volume up", "louder", "turn it up", "turn up the volume", "increase the volume",
                         "make it louder", "raise the volume", "I can't hear it"],
           "volume_down": ["volume down", "quieter", "turn it down", "turn down the volume", "lower the volume",
                           "too loud", "decrease the volume", "make it quieter"],
           "mute": ["mute", "mute the audio", "mute the sound", "silence", "mute the music", "mute everything"]},
}

APPS = {
    "th": {"carplay": ["เปิด CarPlay", "เปิดคาร์เพลย์", "เข้า CarPlay", "ใช้คาร์เพลย์", "เปิดแอปเปิลคาร์เพลย์",
                       "ไปที่ CarPlay", "เปิด Apple CarPlay", "เชื่อมต่อคาร์เพลย์", "เข้าคาร์เพลย์"],
           "android_auto": ["เปิด Android Auto", "เปิดแอนดรอยด์ออโต้", "เข้า Android Auto", "ใช้แอนดรอยด์ออโต้",
                            "ไปที่ Android Auto", "เชื่อมต่อแอนดรอยด์ออโต้", "เข้าแอนดรอยด์ออโต้"]},
    "en": {"carplay": ["open CarPlay", "launch CarPlay", "start CarPlay", "go to CarPlay", "go to Apple CarPlay",
                       "CarPlay", "open Apple CarPlay", "connect CarPlay", "show CarPlay"],
           "android_auto": ["open Android Auto", "launch Android Auto", "start Android Auto", "go to Android Auto",
                            "Android Auto", "connect Android Auto", "show Android Auto"]},
}

# --------------------------------------------------------------------------- none (not a supported control)
NONE = {
    "th": ["โทรหาพ่อ", "โทรหาเพื่อน", "โทรกลับเบอร์ล่าสุด", "นำทางไปสนามบิน", "พาไปปั๊มน้ำมันที่ใกล้ที่สุด",
           "หาร้านกาแฟแถวนี้", "พรุ่งนี้อากาศเป็นยังไง", "ตอนนี้กี่โมง", "วันนี้วันอะไร", "เล่ามุกหน่อย", "สวัสดี",
           "ขอบคุณนะ", "ขับไปอีกกี่กิโล", "รถติดไหม", "อ่านข้อความล่าสุด", "ส่งข้อความหาแฟน", "ตั้งนาฬิกาปลุก",
           "เปิดไฟหน้า", "ล็อกประตู", "บีบแตร", "เปิดไฟฉุกเฉิน", "นำทางกลับบ้าน", "ไปที่ทำงาน", "หาที่จอดรถ",
           "ฝนจะตกไหม", "อืม", "เอ่อ", "ไม่มีอะไร", "ยกเลิก", "ร้องเพลงให้ฟังหน่อย", "เช็คอีเมล", "ข่าววันนี้",
           "เปิดไฟห้องโดยสาร", "ปรับกระจกมองข้าง", "เปิดที่ปัดน้ำฝน", "แบตเหลือกี่เปอร์เซ็นต์", "วิ่งได้อีกกี่กิโล"],
    "en": ["call dad", "call my friend", "call back the last number", "navigate to the airport",
           "find the nearest gas station", "find a coffee shop nearby", "what's the weather today", "what time is it",
           "what day is it", "tell me a joke", "hello", "thank you", "how far is it", "is there traffic",
           "read my messages", "send a message to John", "set an alarm", "turn on the headlights", "lock the doors",
           "honk the horn", "turn on the hazard lights", "take me home", "go to work", "find parking",
           "will it rain", "um", "never mind", "cancel", "sing me a song", "check my email", "what's the news",
           "turn on the cabin light", "adjust the side mirrors", "turn on the wipers", "how much range is left",
           "what's my battery percentage"],
}
