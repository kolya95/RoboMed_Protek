import pandas as pd
import numpy as np
from emoji import emojize
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, \
    ReplyKeyboardMarkup, KeyboardButton, ChatAction, InputMediaPhoto, LabeledPrice
from telegram.ext import Updater, CommandHandler, ConversationHandler, \
    MessageHandler, Filters, RegexHandler, CallbackQueryHandler, PreCheckoutQueryHandler
from pydub import AudioSegment
from google.cloud import speech
import os
import io
import os
from matplotlib import pyplot as plt
import pymystem3
import pymorphy2
import nltk
from nltk.stem.snowball import RussianStemmer
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import brown
morfer = pymorphy2.MorphAnalyzer()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './voicos-ef9646a397b1.json'

TOKEN = # add token

MAIN, CHOICE = range(2)

###
link_table = pd.read_csv('./final_out.csv', index_col=0)
link_table['clear_symptoms'] = link_table['clear_symptoms'].apply(lambda yy: [x[1:-1] for x in yy[1:-1].split(', ')])
link_table['symptoms'] = link_table['symptoms'].apply(lambda yy: [x[1:-1] for x in yy[1:-1].split(', ')])


def add_ngrams(list_of_ends):
    list_of_ngramms = []
    for i in range(len(list_of_ends[-1])):
        list_of_ngramms.append(tuple(list_of_ends[j][i] for j in range(len(list_of_ends))))
    return Counter(list_of_ngramms)


class NGramStorage:
    def add_ngrams(self, list_of_ends):
        list_of_ngramms = []
        for i in range(len(list_of_ends[-1])):
            list_of_ngramms.append(tuple(list_of_ends[j][i] for j in range(len(list_of_ends))))
        return Counter(list_of_ngramms)

    def __init__(self, sents=[], max_n=0):
        self.__max_n = max_n
        self.__ngrams = {i: Counter() for i in range(self.__max_n + 1)}

        # self._ngrams[K] should have the following interface:
        # self._ngrams[K][(w_1, ..., w_K)] = number of times w_1, ..., w_K occured in words
        # self._ngrams[0][()] = number of all words

        self.__ngrams[0][()] = np.sum([len(sents[i]) for i in range(len(sents))])
        tokens = [' '.join(sents[i]) for i in range(len(sents))]
        tokens = (' '.join(tokens)).split(' ')
        list_of_ends = []
        list_of_ends.append(tokens)
        for n in range(1, self.__max_n + 1):
            self.__ngrams[n] = add_ngrams(list_of_ends)
            list_of_ends.append(tokens[n:])

    def ret_ngrams(self):
        return self.__ngrams

    def add_unk_token(self):
        """Add UNK token to 1-grams."""
        # In order to avoid zero probabilites
        if self.__max_n == 0 or 'UNK' in self.__ngrams[1]:
            return
        self.__ngrams[0][()] += 1
        self.__ngrams[1][('UNK',)] = 1

    @property
    def max_n(self):
        """Get max_n"""
        return self.__max_n

    def __getitem__(self, k):
        """Get dictionary of k-gram frequencies.
        Args:
            k (int): length of returning ngrams' frequencies.
        Returns:
            Dictionary (in fact Counter) of k-gram frequencies.
        """
        # Cheking the input
        if not isinstance(k, int):
            raise TypeError('k (length of ngrams) must be an integer!')
        if k > self.__max_n:
            raise ValueError('k (length of ngrams) must be less or equal to the maximal length!')
        return self.__ngrams[k]

    def __call__(self, ngram):
        """Return frequency of a given ngram.
        Args:
            ngram (tuple): ngram for which frequency should be computed.
        Returns:
            Frequency (int) of a given ngram.
        """
        # Cheking the input
        if not isinstance(ngram, tuple):
            raise TypeError('ngram must be a tuple!')
        if len(ngram) > self.__max_n:
            raise ValueError('length of ngram must be less or equal to the maximal length!')
        if len(ngram) == 1 and ngram not in self.__ngrams[1]:
            return self.__ngrams[1][('UNK',)]
        return self.__ngrams[len(ngram)][ngram]


final_list_of_ngrams = []
for ll in link_table.clear_text.values:
    final_list_of_ngrams.append(nltk.word_tokenize(str(ll)))
storage = NGramStorage(final_list_of_ngrams, 7)
ngrams_diseases = storage.ret_ngrams()

final_counter_diseases = Counter()
for i in range(1, 8):
    fff = ngrams_diseases[i]
    for obj_i in fff.most_common(1000):
        if (',' in obj_i[0]) or ('.' in obj_i[0]) or ('(' in obj_i[0]) or (')' in obj_i[0]) \
                or ('-' in obj_i[0]) or ('\'' in obj_i[0]) or (':' in obj_i[0]) or (';' in obj_i[0]):
            pass
        else:
            current_obj_i = ' '.join(list(obj_i[0]))
            final_counter_diseases[' '.join(list(obj_i[0]))] += obj_i[1]


def do_morfy(text):
    new_text = ''
    for token in nltk.word_tokenize(text):
        new_text += morfer.parse(token)[0].normal_form + ' '
    if len(new_text) > 0:
        new_text = new_text[:-1]
    return new_text


def noraml_elem_of_diseases(x):
    x = do_morfy(str(x))
    list_of_symptoms = []
    tokens = nltk.word_tokenize(x)
    for i in range(1, 8):
        for j in range(len(x) - i):
            if ' '.join(tokens[j: j + i]) in final_counter_diseases:
                list_of_symptoms.append(
                    (' '.join(tokens[j: j + i]), final_counter_diseases[' '.join(tokens[j: j + i])]))
    for_return = sorted(list(np.unique([x[0] for x in list_of_symptoms])), key=lambda x: len(x), reverse=True)
    return for_return[:int(len(for_return) / 2) + 1]


def what_disease(link_table, jaloba_mine):
    max_score = 0
    max_disease = ''
    for i in range(link_table.shape[0]):
        current_disease = link_table.iloc[i].disease
        set_symptoms = set(link_table.iloc[i].clear_symptoms)
        set_jaloba = set(jaloba_mine)
        inters = set_jaloba.intersection(set_symptoms)
        score = len(inters) / len(set_jaloba)
        if (score > max_score) & (len(set_symptoms) > 4):
            max_score = score
            max_disease = current_disease
    if score < 0.1:
        score *= 5
    if score > 0.8:
        score = 0.75
    return score, max_disease
###
print('Ready')


HOW_IT_WORKS = emojize(':mag: Как это работает', use_aliases=True)
ABOUT_US = emojize(':rocket: О нас', use_aliases=True)
RANDOM_DRUG = emojize(':crying_cat_face: Пример жалобы', use_aliases=True)
MAIN_MENU_BUTTON = emojize(':seat: Главное меню', use_aliases=True)
main_menu_keyboard = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text=RANDOM_DRUG)],
                                                   [KeyboardButton(text=HOW_IT_WORKS), KeyboardButton(text=ABOUT_US)]],
                                                 resize_keyboard=True)
back_to_menu_keyboard = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text=MAIN_MENU_BUTTON)]],resize_keyboard=True)


def all_work(bot, update, jaloba_mine):
    bot.send_chat_action(chat_id=update.message.chat_id, action=ChatAction.TYPING)

    try:
        msg = 'Подождите, пожалуйста, сейчас информация обрабатывается'
        bot.send_message(chat_id=update.message.chat_id, text=msg)

        jaloba_mine = noraml_elem_of_diseases(jaloba_mine)
        score, result = what_disease(link_table, jaloba_mine)
        current_tabel = link_table[link_table['disease'] == result]
        disease = current_tabel.disease.values[0]
        fealings = current_tabel.fealings.values[0]
        thinking = current_tabel.thinking.values[0]
        symptoms = current_tabel.symptoms.values[0]

        if score == 0.0:
            score = "<0.1"
        text = "Ваш ближайший диагноз по описанию: {} с вероятностью {}\n".format(disease, score)
        text += "Наблюдается следущая симптоматика:\n"
        for i, t in enumerate(symptoms):
            text += str(i + 1) + ') ' + t + '\n'
        # text += symptoms
        text += '\n'
        try:
            np.isnan(fealings)
        except:
            text += "Сейчас у вас в голове следущие мысли : "
            text += fealings + '\n'
            text += "Все будет нормально.\nПо дороге к врачу убеждайте себя и повторяйте, что: "
            text += thinking

        bot.send_message(chat_id=update.message.chat_id, text=text)

        return MAIN

    except:
        msg = emojize('Что-то пошло не так :confused:', use_aliases=True)
        bot.send_message(chat_id=update.message.chat_id, text=msg)
        return MAIN


def main_voice(bot, update, user_data):
    bot.getFile(update.message.voice.file_id).download('voice.ogg')

    audio = AudioSegment.from_file('voice.ogg', format="ogg")
    audio.export("audio2.raw", format="raw")

    speech_client = speech.Client()
    with io.open('audio2.raw', 'rb') as audio_file:
        content = audio_file.read()
        sample = speech_client.sample(
            content,
            encoding='LINEAR16',
            sample_rate_hertz=48000)
    try:
        alternatives = sample.recognize('ru-RU')
    except Exception as inst:
        bot.sendMessage(update.message.chat_id, str(inst))

    # for alternative in alternatives:
    #     bot.sendMessage(update.message.chat_id, alternative.transcript)

    drug = alternatives[0].transcript
    msg = 'Я распознал вашу речь как ' + drug
    bot.sendMessage(update.message.chat_id, msg)

    return all_work(bot, update, drug)


def main(bot, update, user_data):
    jaloba_mine = update.message.text

    return all_work(bot, update, jaloba_mine)


def start(bot, update, user_data):
    msg = 'Добрый день, я помощник по определению болезни по сырому тексту жалобы. ' \
          'Введите жалобу в свободном формате (желательно, без ошибок в орфографии)!'
    bot.send_message(chat_id=update.message.chat_id, text=msg, reply_markup=main_menu_keyboard)

    return MAIN


def random_drug(bot, update):
    msg = 'Ловите пример жалобы для теста работы'
    bot.send_message(chat_id=update.message.chat_id, text=msg)
    msg = 'Жалуюсь на повышение температуры, общую слабость, отёчность и боль при абсцессе'
    bot.send_message(chat_id=update.message.chat_id, text=msg, reply_markup=main_menu_keyboard)
    return MAIN


def print_about_us(bot, update):
    msg = 'Мы большие молодцы! ФКН ТОП!'
    bot.send_message(chat_id=update.message.chat_id, text=msg, reply_markup=main_menu_keyboard)
    return MAIN


def print_how_it_works(bot, update):
    msg = 'Из симптомов диагноза извлекается список диагнозов, ' \
          'из жалобы (сырой текст) вытаскиваются ооооочень умно (Ngramms, нормализация слов, умная токинизация) ' \
          'тоже список симптомов и вспомогательные переменные, которые помогают найти ближайший диагноз'
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
    about_us_handler = RegexHandler(ABOUT_US, print_about_us)
    how_it_works_handler = RegexHandler(HOW_IT_WORKS, print_how_it_works)
    main_button_handler = MessageHandler(Filters.text, main, pass_user_data=True)
    voice_button_handler = MessageHandler(Filters.voice, main_voice, pass_user_data=True)
    fall_entry = [start_handler, random_drug_handler, about_us_handler, how_it_works_handler,
                  voice_button_handler, main_button_handler]

    conv_handler = ConversationHandler(
        entry_points=fall_entry,

        states={
            MAIN: [RegexHandler(RANDOM_DRUG, random_drug),
                   RegexHandler(ABOUT_US, print_about_us),
                   RegexHandler(HOW_IT_WORKS, print_how_it_works),
                   MessageHandler(Filters.voice, main_voice, pass_user_data=True),
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
