from slackclient import SlackClient


def slack_message(message, channel, token):
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel, text=message, username='Results', icon_emoji=':robot_face:')
