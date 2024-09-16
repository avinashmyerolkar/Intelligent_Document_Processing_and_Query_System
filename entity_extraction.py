import json
from pdfminer.high_level import extract_text
from dotenv import dotenv_values

config = dotenv_values('config.env')
gpt_api_key = config['OPENAI_API_KEY']

SYSTEM_PROMPT = """You are a smart and intelligent Named Entity Recognition (NER) system. I will provide you the definition of the entities you need to extract, the sentence from where your extract the entities and the output format with examples.".
Refer to the details mentioned below inorder to extracted text.
    "1. Equipment_name: Name of a equipment like Smart TV, Robotic Vacuum Cleaner, Washing Machine etc.\n"
    "2. Domain: Domain can be only among electronics, mechanical and software.\n"
    "3. Model_numbers: Name of models like VM-O55 (OLED), CB-100 (Basic), CP-FL800 (Front Load) etc.\n"
    "4. Manufacturer: Name of manufacturer, like ViewMax, CleanBot, CleanPro etc.\n"

        Give the output in JSON object which will contain only the mentioned categories like "Equipment_name", "Domain", "Model_numbers", "Manufacturer".
        If there is no text present for which belongs to the particular categories then return empty list.Refer below examples for your understanding.
""" 

USER_PROMPT_1 = """
CleanBot Robotic Vacuum Cleaner FAQ 
1. Product Overview 
CleanBot offers a range of robotic vacuum cleaners to suit different cleaning needs. Our 
current lineup includes: 
• CB-100 (Basic) 
• CB-200 (Smart Navigation) 
• CB-300 (Self-Emptying) 
Each model is designed to provide efficient and hassle-free cleaning. 
2. Technical Specifications 
CB-100 (Basic) 
Battery Life: 90 minutes 
Suction Power: 2000Pa 
Dustbin Capacity: 0.5L 
Noise Level: 60dB 
Weight: 3.5kg 
CB-200 (Smart Navigation) 
Battery Life: 90 minutes 
Suction Power: 2000Pa 
Dustbin Capacity: 0.5L 
Noise Level: 60dB 
Weight: 3.5kg 
CB-300 (Self-Emptying) 
Battery Life: 90 minutes 
Suction Power: 2000Pa 
Dustbin Capacity: 0.5L 
Noise Level: 60dB 
Weight: 3.5kg 
3. Key Features 
• Efficient Cleaning: High suction power and multiple cleaning modes. 
• Smart Navigation: Advanced sensors for better navigation and cleaning. 
• Self-Emptying: Automatically empties its dustbin for hands-free maintenance. 
4. Setup and Installation 
Step-by-step instructions for setting up your CleanBot robotic vacuum cleaner: 
1. Unbox the vacuum cleaner and charging dock. 
2. Place the charging dock against a wall and plug it in. 
3. Place the vacuum cleaner on the dock to charge. 
4. Download the CleanBot app and follow the setup instructions. 
5. Usage Instructions 
Detailed guidelines on using various features and optimizing performance: 
• Use the app to start, stop, and schedule cleaning sessions. 
• Select different cleaning modes (e.g., auto, spot, edge) based on your needs. 
• Monitor the cleaning progress and battery status through the app. 
6. Maintenance and Care 
Guidelines for keeping your robotic vacuum cleaner in top condition: 
• Regularly empty the dustbin and clean the filters. 
• Check and clean the brushes and sensors to ensure optimal performance. 
• Keep the firmware updated for enhanced features and performance. 
7. Troubleshooting 
Common issues and their solutions: 
Problem: Vacuum Cleaner Not Charging 
Solution: Ensure the charging dock is plugged in and the vacuum is properly aligned with 
the dock. 
Problem: Reduced Suction Power 
Solution: Check for blockages in the suction path and clean the filters. 
Problem: Random Navigation 
Solution: Reset the navigation system through the app and ensure the sensors are clean. 
8. Warranty Information 
Comprehensive warranty details for your CleanBot robotic vacuum cleaner: 
• Coverage period: 1 year from the date of purchase. 
• Includes free repair and replacement for manufacturing defects. 
• Excludes damage caused by misuse or unauthorized modifications. 
9. Customer Support 
Information on how to get support for your CleanBot robotic vacuum cleaner: 
• Visit our online support portal for FAQs and troubleshooting guides. 
• Contact our support team via phone or email for personalized assistance. 
• Locate the nearest service center for in-person support.
"""

ASSISTANT_PROMPT_1 = """
            {"Equipment_name": ["Vacuum Cleaner"],
             "Domain": ["electronics"],
             "Model_numbers": ["CB-100","CB-200","CB-300"],
             "Manufacturer": ["CleanBot Robotic"]
            }"""

USER_PROMPT_2 = """
CompuTech Laptop FAQ 
1. Product Overview 
CompuTech offers a range of laptops to meet diverse computing needs. Our current lineup 
includes: 
• CT-B500 (Business) 
• CT-G700 (Gaming) 
• CT-U300 (Ultralight) 
Each model is engineered with high-performance components and innovative features to 
enhance user experience. 
2. Technical Specifications 
CT-B500 (Business) 
Processor: Intel i7 
Memory: 16GB RAM 
Storage: 512GB SSD 
Display: 15.6-inch Full HD 
Graphics: NVIDIA GTX 1650 
Battery Capacity: 6000mAh 
CT-G700 (Gaming) 
Processor: Intel i7 
Memory: 16GB RAM 
Storage: 512GB SSD 
Display: 15.6-inch Full HD 
Graphics: NVIDIA GTX 1650 
Battery Capacity: 6000mAh 
CT-U300 (Ultralight) 
Processor: Intel i7 
Memory: 16GB RAM 
Storage: 512GB SSD 
Display: 15.6-inch Full HD 
Graphics: NVIDIA GTX 1650 
Battery Capacity: 6000mAh 
3. Key Features 
• High-Performance Processor: Ensures smooth and fast computing experience for all tasks. 
• Advanced Graphics: Provides stunning visuals for gaming and multimedia. 
• Long Battery Life: Offers extended usage without frequent recharging. 
4. Setup and Installation 
Step-by-step instructions for setting up your new CompuTech laptop: 
1. Unbox the laptop and connect the charger. 
2. Power on the laptop and follow the on-screen setup instructions. 
3. Install essential software and updates. 
5. Usage Instructions 
Detailed guidelines on using various features and optimizing performance: 
• Use the control panel to manage hardware settings. 
• Customize the desktop environment for better productivity. 
• Utilize power-saving modes to extend battery life. 
6. Maintenance and Care 
Guidelines for keeping your laptop in top condition: 
• Regularly clean the keyboard and screen. 
• Ensure proper ventilation to prevent overheating. 
• Keep the software updated to enhance security and performance. 
7. Troubleshooting 
Common issues and their solutions: 
Problem: Slow Performance 
Solution: Close unnecessary applications and check for malware. 
Problem: Overheating 
Solution: Ensure proper ventilation and clean the cooling fans. 
Problem: Battery Not Charging 
Solution: Check the charger and battery health. Replace if necessary. 
8. Warranty Information 
Comprehensive warranty details for your CompuTech laptop: 
• Coverage period: 2 years from the date of purchase. 
• Includes free repair and replacement for manufacturing defects. 
• Excludes damage caused by misuse or unauthorized modifications. 
9. Customer Support 
Information on how to get support for your CompuTech laptop: 
• Visit our online support portal for FAQs and troubleshooting guides. 
• Contact our support team via phone or email for personalized assistance. 
• Locate the nearest service center for in-person support.
"""

ASSISTANT_PROMPT_2 = """
           {"Equipment_name": ["Laptop"],
            "Domain": ["software"],
            "Model_numbers": ["CT-B500","CT-G700","CT-U300"],
            "Manufacturer": ["CompuTech"]
            }"""

FEW_SHOT_ANSWER_2 ="""
{"Equipment_name": ["सरे गुड आफ्टरनून अमीत दिललीप दल विज जी से बात हो रही है मेरी ।",
                                    "सर  मैं यश बैंक के बिहाज़ से नहया बात कर रही हूँ"],
            "Active Listening": ["क्या आप दोबारा बोल सकते हैं"],
            "Professional": [],
            "Language and Grammar": [],
            "Telephone Etiquettes & Hold Procedure": [],
            "Thanking the Customer": ["थैंक यू आपका कीमती समय देने क लिए"]
            }
            """

import openai
openai.api_key = gpt_api_key
def openai_chat_completion_response(final_prompt):
    response = openai.ChatCompletion.create(
              model="gpt-4o",
              messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_1},
                    {"role": "assistant", "content": ASSISTANT_PROMPT_1},
                    {"role": "user", "content": USER_PROMPT_2},
                    {"role": "assistant", "content": ASSISTANT_PROMPT_2},
                    {"role": "user", "content": final_prompt},
                     ],
              api_key = gpt_api_key
                    )
    json_string = response['choices'][0]['message']['content'].strip(" \n")
    try:
        parsed_dict = json.loads(json_string)
    except:
        start_index = json_string.find("{")
        end_index = json_string.rfind("}") + 1
        json_dict_string = json_string[start_index:end_index]
        parsed_dict = json.loads(json_dict_string)

    return parsed_dict


my_sentence = """CleanBot Robotic Vacuum Cleaner FAQ 
1. Product Overview 
CleanBot offers a range of robotic vacuum cleaners to suit different cleaning needs. Our 
current lineup includes: 
• CB-100 (Basic) 
• CB-200 (Smart Navigation) 
• CB-300 (Self-Emptying) 
Each model is designed to provide efficient and hassle-free cleaning. 
2. Technical Specifications 
CB-100 (Basic) 
Battery Life: 90 minutes 
Suction Power: 2000Pa 
Dustbin Capacity: 0.5L 
Noise Level: 60dB 
Weight: 3.5kg 
CB-200 (Smart Navigation) 
Battery Life: 90 minutes 
Suction Power: 2000Pa 
Dustbin Capacity: 0.5L 
Noise Level: 60dB 
Weight: 3.5kg 
CB-300 (Self-Emptying) 
Battery Life: 90 minutes 
Suction Power: 2000Pa 
Dustbin Capacity: 0.5L 
Noise Level: 60dB 
Weight: 3.5kg 
3. Key Features
• Efficient Cleaning: High suction power and multiple cleaning modes. 
• Smart Navigation: Advanced sensors for better navigation and cleaning. 
• Self-Emptying: Automatically empties its dustbin for hands-free maintenance. 
4. Setup and Installation 
Step-by-step instructions for setting up your CleanBot robotic vacuum cleaner: 
1. Unbox the vacuum cleaner and charging dock. 
2. Place the charging dock against a wall and plug it in. 
3. Place the vacuum cleaner on the dock to charge. 
4. Download the CleanBot app and follow the setup instructions. 
5. Usage Instructions 
Detailed guidelines on using various features and optimizing performance: 
• Use the app to start, stop, and schedule cleaning sessions. 
• Select different cleaning modes (e.g., auto, spot, edge) based on your needs. 
• Monitor the cleaning progress and battery status through the app. 
6. Maintenance and Care 
Guidelines for keeping your robotic vacuum cleaner in top condition: 
• Regularly empty the dustbin and clean the filters. 
• Check and clean the brushes and sensors to ensure optimal performance. 
• Keep the firmware updated for enhanced features and performance. 
7. Troubleshooting 
Common issues and their solutions: 
Problem: Vacuum Cleaner Not Charging 
Solution: Ensure the charging dock is plugged in and the vacuum is properly aligned with 
the dock. 
Problem: Reduced Suction Power 
Solution: Check for blockages in the suction path and clean the filters. 
Problem: Random Navigation 
Solution: Reset the navigation system through the app and ensure the sensors are clean. 
8. Warranty Information 
Comprehensive warranty details for your CleanBot robotic vacuum cleaner: 
• Coverage period: 1 year from the date of purchase. 
• Includes free repair and replacement for manufacturing defects. 
• Excludes damage caused by misuse or unauthorized modifications. 
9. Customer Support 
Information on how to get support for your CleanBot robotic vacuum cleaner: 
• Visit our online support portal for FAQs and troubleshooting guides. 
• Contact our support team via phone or email for personalized assistance. 
• Locate the nearest service center for in-person support. 
"""



if __name__=="__main__":
    #file_path = "/home/avinash_m_yerolkar/frontend/data/ViewMax_Smart_TV_FAQ.pdf"
    #text_extracted = entity_extractor(pdf_path=file_path)
    result = openai_chat_completion_response(final_prompt=my_sentence)
    print(result)



