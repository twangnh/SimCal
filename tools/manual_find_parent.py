import pickle

train_info = pickle.load(open('./lvis_train_cats_info.pt', 'rb'))
val_info = pickle.load(open('./lvis_val_cats_info.pt', 'rb'))
train_on_val_cls_lessthan_100ins_morethan_10ins = [train_item for idx, (train_item, val_item) in enumerate(zip(train_info, val_info))
                    if val_item['instance_count']>0 and train_item['instance_count']>=10 and train_item['instance_count']<100]

train_on_val_cls_less_than_10ins = [train_item for idx, (train_item, val_item) in enumerate(zip(train_info, val_info))
                    if val_item['instance_count']>0 and train_item['instance_count']<10]


# >=10 and <100 cls parent class
['aerosol_can', 'alarm_clock', 'anklet', 'aquarium', 'armband', 'armor', 'ashtray', 'atomizer', 'basketball_backboard', 'bandage', 'barrette', 'barrow', 'basketball_hoop', 'basketball', 'bathrobe', 'battery', 'beanbag',
 'bib', 'birdfeeder', 'birdcage', 'black_sheep', 'blazer', 'boiled_egg',
 'deadbolt', 'bookcase', 'bouquet', 'bowler_hat', 'bowling_ball', 'suspenders', 'brassiere', 'bread-bin', 'bridal_gown',
 'briefcase', 'bristle_brush', 'broom', 'brownie', 'horse_buggy',
 'bullet_train', 'bunk_bed',
 'burrito', 'calculator', 'camcorder', 'camper_(vehicle)', 'candy_cane', 'walking_cane', 'canoe', 'cantaloup', 'cape', 'cappuccino', 'identity_card', 'horse_carriage', 'cart', 'carton', 'cash_register', 'cayenne_(spice)', 'CD_player', 'chap', 'chocolate_bar', 'chocolate_cake', 'slide', 'cigarette_case', 'cleansing_agent', 'clementine', 'clipboard', 'clothes_hamper', 'clothespin', 'coatrack', 'cock', 'colander', 'coleslaw', 'pacifier', 'cookie_jar', 'costume', 'cowbell', 'crab_(animal)', 'crape', 'crayon', 'crib', 'crow', 'crown', 'cruise_ship', 'police_cruiser', 'cub_(animal)', 'trophy_cup', 'hair_curler', 'cutting_tool', 'deer', 'detergent', 'diaper', 'die', 'dish_antenna', 'domestic_ass', 'doormat', 'underdrawers', 'dress_hat', 'drum_(musical_instrument)', 'dumpster', 'eagle', 'easel', 'eclair', 'egg_yolk', 'eggbeater', 'eraser', 'Ferris_wheel', 'ferry', 'fighter_jet', 'file_cabinet', 'fire_engine', 'fire_hose', 'fish_(food)', 'fishbowl', 'fishing_rod', 'flashlight', 'flute_glass', 'football_(American)', 'footstool', 'forklift', 'freight_car', 'freshener', 'frog', 'fruit_juice', 'garbage', 'garbage_truck', 'garden_hose', 'gargoyle', 'gelatin', 'giant_panda', 'ginger', 'cincture', 'globe', 'golfcart', 'grater', 'grizzly', 'grocery_bag', 'hairpin', 'hammer', 'handkerchief', 'veil', 'headscarf', 'headset', 'heart', 'heater', 'helicopter', 'highchair', 'hockey_stick', 'hog', 'honey', 'hot_sauce', 'icecream', 'igniter', 'iron_(for_clothing)', 'jeep', 'jewelry', 'jumpsuit', 'kayak', 'kettle', 'kitchen_table', 'kitten', 'lab_coat', 'ladybug', 'lip_balm', 'lipstick', 'liquor', 'lizard', 'lollipop', 'loveseat', 'mallet', 'manger', 'mashed_potato', 'mat_(gym_equipment)', 'measuring_stick', 'melon', 'mixer_(kitchen_tool)', 'monkey', 'mouse_(animal_rodent)', 'nameplate', 'nightshirt', 'noseband_(for_animals)', 'notepad', 'oil_lamp', 'olive_oil', 'omelet', 'ostrich', 'overalls_(clothing)', 'owl', 'packet', 'paintbrush', 'palette', 'parachute', 'parakeet', 'parrot', 'peanut_butter', 'pelican', 'pepper_mill', 'perfume', 'phonograph_record', 'piano', 'pita_(bread)', 'platter', 'playing_card', 'pliers', 'pony', 'postbox_(public)', 'potholder', 'pottery', 'projectile_(weapon)', 'projector', 'puppy', 'racket', 'radio_receiver', 'ram_(animal)', 'reamer_(juicer)', 'receipt', 'recliner', 'rhinoceros', 'rocking_chair', 'roller_skate', 'rolling_pin', 'router_(computer_equipment)', 'salsa', 'saucepan', 'school_bus', 'scoreboard', 'scrambled_eggs', 'screwdriver', 'scrubbing_brush', 'shaving_cream', 'shield', 'shopping_cart', 'shot_glass', 'shovel', 'silo', 'sled', 'sleeping_bag', 'slipper_(footwear)', 'snowman', 'solar_array', 'soupspoon', 'sour_cream', 'spice_rack', 'sponge', 'stapler_(stapling_machine)', 'steak_(food)', 'step_stool', 'stereo_(sound_system)', 'stockings_(leg_wear)', 'strainer', 'mop', 'sweatband', 'sword', 'table_lamp', 'tape_measure', 'teakettle', 'telephone_booth', 'thermos_bottle', 'tiara', 'tights_(clothing)', 'timer', 'tinsel', 'toast_(food)', 'toaster_oven', 'toolbox', 'cover', 'tortilla', 'tow_truck', 'tractor_(farm_equipment)', 'dirt_bike', 'tripod', 'turban', 'turkey_(bird)', 'turkey_(food)', 'turtle', 'underwear', 'vacuum_cleaner', 'valve', 'vending_machine', 'volleyball', 'waffle', 'wagon', 'walking_stick', 'wall_clock', 'wallet', 'automatic_washer', 'water_faucet', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'webcam', 'wheelchair', 'wig', 'wind_chime', 'windsock', 'wine_bucket', 'wok', 'wreath', 'yak', 'yogurt']



{'mandarin_orange': 'orange',
 'shot_glass': 'glass',
 'hourglass': 'glass',
 'flute_glass': 'glass',
 'salad_plate': 'plate',
 'black_sheep': 'sheep',
 'blazer': 'suit',
 'briefcase': 'suitcase',
 'bouquet': 'flower_arrangement',
 'bowler_hat': 'hat',
 'sunhat': 'hat',
 'dress_hat': 'hat',
 'cowboy_hat': 'hat',
 'bowling_ball': 'ball',
 'basketball':'ball',
 'volleyball':'ball',
 'softball':'ball',
 'beachball':'ball',
 'ping-pong_ball':'ball',
 'paintbrush':'pen',
 'bristle_brush': 'hairbrush',
 'chocolate_cake': 'cake',
 'horse_buggy': 'horse_carriage',## this two may combine
 'bullet_train': 'train',
 'bunk_bed':'bed',

 }


'ball, hat'
'different clothes:, eg, brassiere, bridal gown'


## lessthan 10 not in val num227
['acorn', 'apple_juice', 'applesauce', 'bagpipe', 'baguet', 'bait', 'ballet_skirt', 'Band_Aid', 'banjo', 'barbell', 'barge',
 'bass_horn', 'bat_(animal)', 'beachball', 'beaker', 'beeper', 'beetle', 'birdbath', 'pirate_flag', 'blimp', 'boar',
 'bolo_tie', 'bonnet', 'bookmark', 'pipe_bowl', 'brass_plaque', 'breechcloth', 'bubble_gum', 'bulldog', 'bulldozer', 'bulletproof_vest', 'candy_bar', 'cannon', 'canteen', 'elevator_car', 'car_battery', 'caviar', 'chain_mail', 'chaise_longue', 'checkbook', 'Chihuahua', 'chime', 'chocolate_milk', 'chocolate_mousse', 'cigar_box', 'clarinet', 'clutch_bag', 'coffee_filter', 'coil', 'convertible_(automobile)', 'sofa_bed', 'corkboard', 'cornmeal', 'corset', 'cougar', 'curling_iron', 'custard', 'cylinder', 'dagger', 'dartboard', 'diary', 'dishwasher_detergent', 'diskette', 'dollar', 'doorbell', 'dove', 'drinking_fountain', 'drone', 'dropper', 'Dutch_oven', 'earplug', 'eel', 'electric_chair', 'escargot', 'eyepatch', 'falcon', 'fedora', 'ferret', 'file_(tool)', 'fishing_boat', 'fleece', 'football_helmet', 'fruit_salad', 'fudge', 'gag', 'gasmask', 'gemstone', 'gondola_(boat)', 'gorilla', 'grasshopper', 'griddle', 'grillroom', 'grinder_(tool)', 'grits', 'hair_spray', 'hamper', 'hamster', 'hand_glass', 'handcuff', 'hardback_book', 'harmonium', 'hatbox', 'hearing_aid', 'hot-air_balloon', 'hotplate', 'hourglass', 'houseboat', 'hummingbird', 'hummus', 'popsicle', 'ice_pack', 'ice_skate', 'inhaler', 'kennel', 'keycard', 'knight_(chess_piece)', 'knocker_(on_a_door)', 'koala', 'lamb-chop', 'lasagna', 'lawn_mower', 'lemonade', 'lightning_rod', 'limousine', 'linen_paper', 'Loafer_(type_of_shoe)', 'machine_gun', 'mammoth', 'mascot', 'masher', 'microscope', 'milestone', 'music_stool', 'nailfile', 'neckerchief', 'nosebag_(for_animals)', 'nutcracker', 'octopus_(food)', 'octopus_(animal)', 'oregano', 'inkpad', 'paintbox', 'paperweight', 'passenger_ship', 'patty_(food)', 'pegboard', 'pencil_box', 'pencil_sharpener', 'pendulum', 'persimmon', 'phonebook', 'ping-pong_ball', 'pinwheel', 'pistol', 'pitchfork', 'portrait', 'power_shovel', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'puncher', 'puppet', 'quesadilla', 'race_car', 'radar', 'rag_doll', 'rat', 'record_player', 'red_cabbage', 'river_boat', 'road_map', 'root_beer', 'safety_pin', 'salad_plate', 'salmon_(food)', 'saxophone', 'scarecrow', 'scraper', 'seaplane', 'seedling', 'sharpener', 'shawl', 'shepherd_dog', 'sherbert', 'shredder_(for_paper)', 'sling_(bandage)', 'snake', 'soda_fountain', 'softball', 'soup_bowl', 'soya_milk', 'spear', 'spider', 'squirrel', 'steak_knife', 'steamer_(kitchen_appliance)', 'stew', 'subwoofer', 'sugarcane_(plant)', 'Tabasco_sauce', 'table-tennis_table', 'tachometer', 'army_tank', 'telephoto_lens', 'tequila', 'thimble', 'tree_house', 'triangle_(musical_instrument)', 'truffle_(chocolate)', 'vat', 'turtleneck_(clothing)', 'typewriter', 'unicycle', 'violin', 'vodka', 'waffle_iron', 'walrus', 'wardrobe', 'wasabi', 'water_filter', 'water_heater', 'water_gun', 'whiskey', 'wing_chair', 'wolf']

{'apple_juice': 'fruit_juice',
 'ballet_skirt':'skirt',
 'banjo': 'guitar',
 'barge': 'cruise_ship',
 'cargo_ship': 'cruise_ship',
 'bat':'bird',
 'beachball': 'ball',
 'beaker': 'glass',
 'beeper': 'cellular telephone',
 'blimp':'airplane',
 'bolo_tie': 'necktie',

 }


## lessthan 10 in val num67
['ax', 'batter_(food)', 'Bible', 'gameboard', 'book_bag', 'boom_microphone', 'bow_(weapon)', 'broach', 'cabin_car',
 'cargo_ship', 'checkerboard', 'chessboard', 'chest_of_drawers_(furniture)', 'comic_book',
 'concrete_mixer', 'dachshund', 'tux', 'dolphin', 'eye_mask', 'dumbbell', 'dustpan',
 'egg_roll', 'elk', 'flash', 'funnel', 'hammock', 'handsaw', 'heron', 'hippopotamus', 'ice_tea', 'incense', 'ironing_board', 'keg', 'martini', 'matchbox', 'mint_candy', 'pantyhose', 'paperback_book', 'peeler_(tool_for_fruit_and_vegetables)', 'piggy_bank', 'tobacco_pipe', 'playpen', 'plow_(farm_equipment)', 'police_van', 'poncho', 'pool_table', 'satchel', 'sawhorse', 'scratcher', 'seahorse', 'sewing_machine', 'shark', 'Sharpie', 'shears', 'sieve', 'carbonated_water', 'space_shuttle', 'stepladder', 'string_cheese', 'sugar_bowl', 'tambourine', 'trampoline', 'tricycle', 'vinegar', 'vulture', 'whistle', 'yoke_(animal_equipment)']

{'Bible': 'book',
 'book_bag': 'backpack',
 'cabin_car':'train',
 'cargo_ship': 'boat',
 'chest_of_drawers_(furniture)': 'cabinet',
 'comic_book': 'book',
 'dachshund': 'dog',
 'dolphin': 'fish',
 'elk': 'deer',
 'heron': 'book',
 'ice_tea': 'cup',
 'incense': 'chopstick',
 'ironing_board': 'table',
 'keg': 'pot',
 'martini': 'wineglass',
 'matchbox':'cigarette_case?', #may find better one
 'mint_candy':'cigarette_case?',#may find better one
 'pantyhose': 'sock',
 'paperback_book': 'book',
 'peeler_(tool_for_fruit_and_vegetables)': 'knife?',#may find better one
 '': '',
 '': '',
 '': '',
 '': '',




 }



mint_candy                                  :
pantyhose
paperback_book                              : book
peeler_(tool_for_fruit_and_vegetables)
piggy_bank
tobacco_pipe
playpen                                     : bed
plow_(farm_equipment)
police_van                                  : car_(automobile)(?)
poncho                                      :
pool_table
satchel                                     : handbag(?)
sawhorse
scratcher
seahorse
sewing_machine
shark                                       : fish
Sharpie                                     : pen
shears                                      : scissors
sieve
carbonated_water                            : bottle
space_shuttle
stepladder                                  : ladder
string_cheese
sugar_bowl                                  : bowl
tambourine
trampoline
tricycle
vinegar                                     : bottle
vulture                                     : bird
whistle
yoke_(animal_equipment)                     :


cake & wedding cake