import jax, time, pickle


class Stats:
    class c:
        CEND = "\33[0m"
        CBLACK = "\33[30m"
        CRED = "\33[31m"
        CGREEN = "\33[32m"
        CYELLOW = "\33[33m"
        CBLUE = "\33[34m"
        CVIOLET = "\33[35m"
        CBEIGE = "\33[36m"
        CWHITE = "\33[37m"

        CYCLE = [CRED, CGREEN, CYELLOW, CBLUE, CVIOLET, CBEIGE, CWHITE]

    class Time:
        @classmethod
        def block(cls, val):
            if type(val).__name__ in ["FrozenDict", "dict"]:
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), val)
            else:
                val.block_until_ready()
            return val

        def __init__(self, key, on_complete_func, **kwargs):
            self.key = key
            self.on_complete_func = on_complete_func
            self.kwargs = kwargs

        def __enter__(self):
            self.t0 = time.time()
            return self.block

        def __exit__(self, *args):
            self.on_complete_func(time.time() - self.t0)

            if self.kwargs.get("print", False):
                print(
                    *[
                        f"{Stats.c.CBLACK}{self.key}",
                        Stats.c.CEND,
                        "\t\t",
                        f"{time.time() - self.t0:.3f}s",
                    ]
                )

    def time(self, key, **kwargs):
        def on_complete(dt):
            for k, v in key.items():
                if not isinstance(v, set):
                    continue

                self.write({k: {next(iter(v)): dt}}, self.dict, **kwargs)

        return Stats.Time(key, on_complete, **kwargs)

    def __init__(self, filename=None):
        self.dict = {}
        self.filename = filename

    def write(self, key_dict_or_str, dict_, **kwargs):
        for k, v in key_dict_or_str.items():
            if isinstance(v, dict):
                if not k in dict_:
                    dict_[k] = {}

                self.write(v, dict_[k], **kwargs)
            else:
                self.write_leaf(k, v, dict_, **kwargs)

    def write_leaf(self, key, value, dict_, append=True, **kwargs):
        if append is not True:
            dict_[key] = value
        elif not key in dict_:
            dict_[key] = [value]
        else:
            dict_[key].append(value)

    def __call__(self, key, **kwrags):
        self.write(key, self.dict, **kwrags)
        return self

    def __repr__(self) -> str:
        return f"Stats{self.dict.__repr__()}"

    def __getitem__(self, key):
        return self.dict[key]

    # def __getattr__(self, name):
    #     print(name)
    #     # call the method on the dict if method
    #     # does not exist on the Stats object
    #     if hasattr(self, name):
    #         return getattr(self, name)

    #     return getattr(self.dict, name)

    def latest(self, *vals, **kwargs):
        acc = []
        for i, val in enumerate(vals):
            i -= 1
            if type(val) is str:
                acc.append(f"{Stats.c.CBLACK}{val}{Stats.c.CEND}")
            elif type(val) is dict:
                c = Stats.c.CYCLE[i % len(Stats.c.CYCLE)]
                for i, (k, v) in enumerate(val.items()):
                    if type(v) is dict:
                        raise Exception("Implement me!")
                    if type(v) is str:
                        v = [v]

                    v = self.dict[k].keys() if v == ["*"] else v
                    a = [f"{c}{k}:["]
                    a.append(
                        " ".join(
                            [
                                f"{c}{vv}: {Stats.c.CWHITE}{self.dict[k][vv][-1]:.3f}{Stats.c.CEND}"
                                for vv in v
                            ]
                        )
                    )
                    a.append(f"{c}]{Stats.c.CEND}")
                    acc.append("".join(a))
            else:
                acc.append(f"<unknown type>")

        return acc

    def persist(self):
        if not self.filename:
            print("*** NOT PERSISTED: No filename provided! ***")
            return None

        with open(self.filename, "wb") as file:
            pickle.dump(self, file)

    def load(self):
        if not self.filename:
            raise Exception("No filename provided!")

        with open(self.filename, "rb") as file:
            self.dict = pickle.load(file).dict

        return self


if __name__ == "__main__":
    stats = Stats("tmp/stats-demo.pkl")

    stats({"top_lvl_append": 32.0})
    stats({"top_lvl_append": 64.0})

    stats({"args": {"this": "that"}}, append=False)
    stats({"train": {"loss": 11}})
    stats({"train": {"loss": 10}})
    stats({"train": {"loss": 9}})
    stats({"train": {"acc": 5}})
    stats({"train": {"acc": 6}})
    stats({"train": {"acc": 7}})
    stats({"test": {"hello": 123}})

    with stats.time({"time": {"load_data"}}, append=False) as block:
        time.sleep(0.256)

    for i in range(8):
        with stats.time({"time": {"train"}}) as block:
            time.sleep(0.032)

    print(
        *stats.latest(
            *[
                f"Stats demo {stats['time']['load_data']:.3f}s",
                f"{stats['time']['train'][-1]:.3f}",
                {"train": "*"},
            ]
        )
    )

    stats.persist()

    stats = Stats("tmp/stats-demo.pkl").load()
