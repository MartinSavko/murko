import redis

class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)
        self.pubsub = self.redis_client.pubsub()
        self.subscribed_channels = set()
        self.is_listening = False

    def _listen_for_messages(self):
        while True:
            message = self.pubsub.get_message()
            if message and message['type'] == 'message':
                print(f"Received message: {message['data'].decode('utf-8')}")

    def set(self, key, value):
        self.redis_client.set(key, value)

    def get(self, key):
        return self.redis_client.get(key)

    def publish(self, channel, message):
        self.redis_client.publish(channel, message)

    def subscribe(self, channel):
        if channel not in self.subscribed_channels:
            self.pubsub.subscribe(channel)
            self.subscribed_channels.add(channel)
            print(f"Subscribed to channel: {channel}")
            if not self.is_listening:
                self.is_listening = True
                self._listen_for_messages()
        else:
            print(f"Already subscribed to channel: {channel}")

    def unsubscribe(self, channel):
        if channel in self.subscribed_channels:
            self.pubsub.unsubscribe(channel)
            self.subscribed_channels.remove(channel)
            print(f"Unsubscribed from channel: {channel}")
        else:
            print(f"Not subscribed to channel: {channel}")

    def close(self):
        self.redis_client.close()


if __name__ == "__main__":
    client = RedisClient()

    # Set and get example
    client.set('metadata', 'test')
    print(client.get('metadata'))

    # Publish/Subscribe example
    client.subscribe('murko')
    client.publish('murko', 'metadata available!')
