#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

from keras.utils import Sequence


def size_differs(original_size, img_size):
    return original_size[0] != img_size[0] or original_size[1] != img_size[1]


def get_dynamic_batch_size(img_size, pixel_budget=768 * 992):
    return max(int(pixel_budget / np.prod(img_size)), 1)


def get_batch(i, img_paths, batch_size):
    half, r = divmod(batch_size, 2)
    indices = np.arange(i - half, i + half + r)
    return [img_paths[divmod(item, len(img_paths))[1]] for item in indices]


def get_dynamic_batch_size(img_size, pixel_budget=768 * 992):
    return max(int(pixel_budget / np.prod(img_size)), 1)


def get_img_size_as_scale_of_pixel_budget(
    scale, pixel_budget=768 * 992, ratio=0.75, modulo=32
):
    n = math.floor(math.sqrt(pixel_budget / ratio))
    new_n = n * scale
    img_size = np.array((new_n * ratio, new_n)).astype(int)
    img_size -= np.mod(img_size, modulo)
    return tuple(img_size)


class JsonDataset(Sequence):
    def __init__(
        self,
        annotations,
        heads,
        batch_size=1,
        dynamic_batch_size=False,
        number_batch_size_scales=32,
        img_size=(256, 320),
        possible_ratios=[0.75, 1.0],
        augment=False,
        swap_backgrounds=True,
        min_scale=0.15,
        max_scale=1.0,
        pixel_budget=768 * 992,
        artificial_size_increase=1,
        shuffle_at_0=False,
        target=True,
        verbose=False,
        workers=10,
        use_multiprocessing=True,
        max_queue_size=10,
    ):

        self.annotations = annotations
        self.heads = heads

        self.batch_size = batch_size
        self.dynamic_batch_size = dynamic_batch_size
        if self.dynamic_batch_size:
            self.batch_size = 1
        self.possible_scales = np.linspace(
            min_scale, max_scale, number_batch_size_scales
        )
        self.img_size = img_size
        self.possible_ratios = possible_ratios

        # augmentation parameters
        self.augment = augment
        self.swap_backgrounds = swap_backgrounds
        self.pixel_budget = pixel_budget
        self.artificial_size_increase = artificial_size_increase
        # if artificial_size_increase > 1:
        # self.annotations = annotations * int(artificial_size_increase)
        # else:

        self.samples = [Sample(item) for item in annotations]
        self.nsamples = len(self.samples)

        if self.swap_backgrounds:
            self.backgrounds = [
                sample
                for sample in self.samples
                if "background" in sample["image_path"]
            ]

        self.shuffle_at_0 = shuffle_at_0
        self.target = target

        self.verbose = verbose

        super().__init__(
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
        )

    def __len__(self):
        return self.nsamples

    def get_empty_sample(self, final_img_size):
        y = []
        for head in self.heads:
            output = np.zeros(final_img_size + (head["channels"],), dtype=head["dtype"])
            y.append(output)
        return y

    def get_empty_batch(self, batch_size, final_img_size):
        y = []
        for head in self.heads:
            output = np.zeros(
                (batch_size,) + final_img_size + (head["channels"],),
                dtype=head["dtype"],
            )
            y.append(output)
        return y

    def get_img_size_and_batch(self):
        if self.dynamic_batch_size:
            img_size = get_img_size_as_scale_of_pixel_budget(
                random.choice(self.possible_scales),
                pixel_budget=self.pixel_budget,
                ratio=random.choice(self.possible_ratios),
            )
            batch_size = get_dynamic_batch_size(
                img_size, pixel_budget=self.pixel_budget
            )
            i = idx
            batch = get_batch(i, self.samples, batch_size)
        else:
            img_size = self.img_size[:]
            batch_size = self.batch_size
            i = idx * self.batch_size
            start_index = i
            end_index = i + batch_size
            batch = self.samples[start_index:end_index]

        return img_size, batch

    def __getitem__(self, idx):
        if idx == 0 and self.shuffle_at_0:
            random.Random().shuffle(self.samples)

        img_size, batch = self.get_img_size_and_batch()
        final_img_size = copy.copy(img_size)
        batch_size = len(batch)

        x = np.zeros((batch_size,) + img_size + (3,), dtype="float32")
        y = self.get_empty_batch(batch_size, final_img_size)

        for j, sample in enumerate(batch):
            x[j], y_j = self.get_sample(sample, final_img_size)
            for k, output in enumerate(y_j):
                y[k][j] = output[k]

        if self.target and len(y) == 1:
            y = y[0]

        return x, y if self.target else x

    def get_sample(self, sample, final_img_size, new_background=None):

        if self.augment:
            if self.swap_backgrounds:
                new_background = random.choice(self.backgrounds)["image"]
            img, points = sample.transform(final_img_size, new_background)
        else:
            img = sample.get_image()
            points = sample.get_points()

        y = []
        for head in self.heads:
            if type(head) != list:
                head = [head]
            for item in head:
                target = sample.get_target(head, img, points)
                y.append(target)

        return img, y
