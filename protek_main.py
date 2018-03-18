import pandas as pd
import numpy as np
from emoji import emojize
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, \
    ReplyKeyboardMarkup, KeyboardButton, ChatAction, InputMediaPhoto, LabeledPrice
from telegram.ext import Updater, CommandHandler, ConversationHandler, \
    MessageHandler, Filters, RegexHandler, CallbackQueryHandler, PreCheckoutQueryHandler

###
dict_reasons = {
    'Упаковка': 1,
    'инструкция турецком языке': 1,

    'Маркировка инструкция': 2,
    'Маркировка ампулах стерта': 2,
    'Маркировка': 2,

    'Отзыв декларации соответствии': 5,
    'Отзыв производителем': 5,
    'Отмена государственной регистрации': 5,
    'Прекращение действия декларации соответствии': 5,
    'Отзыв производителем декларации соответствии': 5,
    'связи преостановлением реализации фармацевтической субстанции': 5,
    'Необходимость гармонизации регистрационного досье нормативной документации': 5,
    'Необходимость гармонизации регистрационных документов': 5,
    'Отмена действия сертификата соответствия': 5,

    'Описание (': 6,
    'Описание,': 6,
    'Цветность': 6,
    'запись отсутствует Государственном реестре': 6,
    'запись о которой отсутствует в Государственном реестре лекарственных средств': 6,

    'Количественное определение': 8,
    'Герметичность': 8,
    'Растворение': 8,

    'Микробиологическая чистота': 10,

    'Фальсификация': 10,
    'Подлинность': 10,
    'Признаки фальсификации': 10,
    'Отмена действия сертификата соответствия': 10,
    'этикетках флаконов отсутствуют серия срок годности': 10,
    'Указание первичной вторичной упаковках': 10,
    'Стабильность препарата': 10,
    'Развитие нежелательных реакций': 10,
    'Растворение': 10,
    'Несоответствие процессе производства': 10,
    'Выявление разночтений документах': 10,
    'Выявлены несоответствия утвержденным нормам спецификации изучения стабильности': 10,
    'Механические включения': 10,
    'Обнаружение частиц полиуретана упаковочном оборудовании': 10,
    'Остаточные органические растворители': 10,

    'Показатель письме указан': 1,
    'Посторонние примеси': 10,
    'Развитие нежелательных реакций': 10
}

med_final = pd.read_csv('med_final.csv', index_col=0)
med_names = pd.read_csv('med_names.csv', index_col=0)
med_prep = pd.read_csv('med_prep.csv', index_col=0)

dict_med_names = {}
for i in range(len(med_names)):
    if med_names.loc[i, 'ТН'] not in dict_med_names:
        dict_med_names[med_names.loc[i, 'ТН']] = med_names.loc[i, 'МНН']

dict_tn_prod = {}
for i in range(len(med_final)):
    if med_final.loc[i, 'ТН'] not in dict_tn_prod:
        dict_tn_prod[med_final.loc[i, 'ТН']] = med_final.loc[i, 'Производитель']

dict_tn_prod_all = {}
for i in range(len(med_prep)):
    if med_prep.loc[i, 'Торговое наименование\nлекарственного препарата'] not in dict_tn_prod_all:
        dict_tn_prod_all[med_prep.loc[i, 'Торговое наименование\nлекарственного препарата']] = med_prep.loc[
            i, 'Юридическое лицо, на имя которого выдано регистрационное удостоверение']

dict_producers = {}
for i in range(len(med_final)):
    if med_final.loc[i, 'Производитель'] not in dict_producers:
        #         print(dict_reasons[med_final.loc[i, 'Причина']])
        dict_producers[med_final.loc[i, 'Производитель']] = dict_reasons[med_final.loc[i, 'Причина']]
    else:
        dict_producers[med_final.loc[i, 'Производитель']] += dict_reasons[med_final.loc[i, 'Причина']]

dict_producers_cnt = {}
for i in range(len(med_final)):
    if med_final.loc[i, 'Производитель'] not in dict_producers_cnt:
        #         print(dict_reasons[med_final.loc[i, 'Причина']])
        dict_producers_cnt[med_final.loc[i, 'Производитель']] = 1
    else:
        dict_producers_cnt[med_final.loc[i, 'Производитель']] += 1


def sorted_producers(medicine_name):
    medicine_names = list(med_names[med_names['МНН'] == medicine_name]['ТН'].unique())
    medicine_names2 = list(med_final[med_final['МНН'] == medicine_name]['ТН'].unique())
    medicine_names += medicine_names2
    rating = []
    for name in medicine_names:
        if name in dict_tn_prod:
            prod = dict_tn_prod[name]
            if prod in dict_producers:
                rating.append((name, prod, np.round(dict_producers[prod] / dict_producers_cnt[prod], 2)))
        elif name in dict_tn_prod_all:
            prod = dict_tn_prod_all[name]
            rating.append((name, prod, 0))
        else:
            rating.append((name, '-', 0))
    rating = sorted(rating, key=lambda x: x[2])
    return rating


def all_rating():
    rating = []
    for name in dict_producers.keys():
        if name in dict_producers_cnt:
            rating.append((name, np.round(dict_producers[name]/dict_producers_cnt[name], 2)))
    rating = sorted(rating, key=lambda x: x[1])
    return rating


top_producers = all_rating()
print('Ready')
###

TOKEN = # add token

MAIN, CHOICE = range(2)

# data = pd.read_csv('med_df_text.csv', usecols=['ТН', 'Производитель'])
# data.dropna(subset=['ТН', 'Производитель'], inplace=True)
# data.drop_duplicates(inplace=True)
# data.reset_index(inplace=True, drop=True)
# data = data[['ТН', 'Производитель']]

HOW_IT_WORKS = emojize(':mag: Как это работает', use_aliases=True)
ABOUT_US = emojize(':rocket: О нас', use_aliases=True)
RANDOM_DRUG = emojize(':pill: Тестовое лекарство', use_aliases=True)
TOP_PRODUCERS = emojize(':star: Лучшие производители', use_aliases=True)
MAIN_MENU_BUTTON = emojize(':seat: Главное меню', use_aliases=True)
main_menu_keyboard = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text=TOP_PRODUCERS)], [KeyboardButton(text=RANDOM_DRUG)],
                                                   [KeyboardButton(text=HOW_IT_WORKS), KeyboardButton(text=ABOUT_US)]],
                                                 resize_keyboard=True)
back_to_menu_keyboard = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text=MAIN_MENU_BUTTON)]],resize_keyboard=True)


def all_work(bot, update, drug):
    bot.send_chat_action(chat_id=update.message.chat_id, action=ChatAction.TYPING)

    try:
        msg = 'Подождите, пожалуйста, сейчас информация обрабатывается'
        bot.send_message(chat_id=update.message.chat_id, text=msg)

        current_drug = sorted_producers(drug)

        msg = ''
        if len(current_drug) == 1:
            msg = 'Only one producer: ' + current_drug[0][0]
        else:
            msg = 'There are several producers:'
            for k in range(len(current_drug)):
                msg += '\n' + str(k + 1) + '. ' + current_drug[k][0]

        bot.send_message(chat_id=update.message.chat_id, text=msg)

        return MAIN

    except:
        msg = emojize('Что-то пошло не так :confused:', use_aliases=True)
        bot.send_message(chat_id=update.message.chat_id, text=msg)
        return MAIN


def main(bot, update, user_data):
    drug = update.message.text

    return all_work(bot, update, drug)


def start(bot, update, user_data):
    msg = 'Вы можете отправить мне название лекарства, а я выдам самых качественных производителей. ' \
          'Также, я могу показать список самых надежных производителей! \n\nПример лекарства:'
    bot.send_message(chat_id=update.message.chat_id, text=msg)
    msg = 'Валерианы настойка'
    bot.send_message(chat_id=update.message.chat_id, text=msg, reply_markup=main_menu_keyboard)

    return MAIN


def random_drug(bot, update):
    msg = 'Ловите лекарство из нашей базы для теста работы'
    bot.send_message(chat_id=update.message.chat_id, text=msg)
    # n = np.random.random_integers(1, data.shape[0]) - 1
    # msg = data['ТН'][n]
    msg = 'Дротаверин'
    bot.send_message(chat_id=update.message.chat_id, text=msg, reply_markup=main_menu_keyboard)
    return MAIN


def print_about_us(bot, update):
    msg = 'Мы большие молодцы! ФКН ТОП!'
    bot.send_message(chat_id=update.message.chat_id, text=msg, reply_markup=main_menu_keyboard)
    return MAIN


def print_how_it_works(bot, update):
    msg = 'Вот как это работает!'
    bot.send_message(chat_id=update.message.chat_id, text=msg, reply_markup=main_menu_keyboard)
    return MAIN


def print_top_producers(bot, update):
    msg = 'ТОП производителей таков:'
    temp = top_producers
    for k in range(10):
        msg += '\n' + str(k+1) + '. ' + temp[k][0]
    bot.send_message(chat_id=update.message.chat_id, text=msg, reply_markup=main_menu_keyboard)
    return MAIN


def setup(webhook_url=None):
    global service
    '''If webhook_url is not passed, run with long-polling.'''
    updater = Updater(TOKEN)
    bot = updater.bot
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start, pass_user_data=True)
    random_drug_handler = RegexHandler(RANDOM_DRUG, random_drug)
    top_producers_handler = RegexHandler(TOP_PRODUCERS, print_top_producers)
    about_us_handler = RegexHandler(ABOUT_US, print_about_us)
    how_it_works_handler = RegexHandler(HOW_IT_WORKS, print_how_it_works)
    main_button_handler = MessageHandler(Filters.text, main, pass_user_data=True)
    fall_entry = [start_handler, top_producers_handler, random_drug_handler, about_us_handler, how_it_works_handler,
                  main_button_handler]

    conv_handler = ConversationHandler(
        entry_points=fall_entry,

        states={
            MAIN: [RegexHandler(RANDOM_DRUG, random_drug),
                   RegexHandler(ABOUT_US, print_about_us),
                   RegexHandler(HOW_IT_WORKS, print_how_it_works),
                   MessageHandler(Filters.text, main, pass_user_data=True)],

            CHOICE: [RegexHandler(MAIN_MENU_BUTTON, main, pass_user_data=True)],

        },
        fallbacks=fall_entry,
        allow_reentry=True
    )

    dispatcher.add_handler(conv_handler)


    # dispatcher.add_handler(start_handler)
    # dispatcher.add_handler(main_button_handler)

    if webhook_url:
        updater.start_webhook(listen='0.0.0.0',
                              port=8443,
                              url_path=TOKEN,
                              key='private.key',
                              cert='cert.pem',
                              webhook_url='https://82.202.246.210:8443/' + TOKEN)
    else:
        bot.set_webhook()
        updater.start_polling()
        updater.idle()


if __name__ == '__main__':
    setup('')
