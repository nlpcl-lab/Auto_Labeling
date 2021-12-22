import wikipediaapi
import argparse
import logging
import tqdm

class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def num_there(s):
    return not any(i.isdigit() for i in s)

def preprocess_1(target: str):
    article = ['a ','the ','an ']
    for a in article:
        if target.startswith(a):
            return target[len(a):]
    return None

def preprocess_2(target: str):
    if ',' in target:
        return target.split(',')[0]
    return None


def preprocess_3(target: str):
    if ':' in target: return False
    prefix = ['Wikipedia','Wikidata','Pages','Articles','All','Short description','Use American','Use mdy','Webarchive']
    for p in prefix:
        if target.startswith(p): return False
        if target.endswith(p): return False
    return num_there(target)


def print_sections(sections, level=0):
    for s in sections:
        logging.info("%s: %s - %s" % ("*" * (level + 1), s.title, s.text[0:40]))
        print_sections(s.sections, level + 1)

def print_categories(page):
    logging.info("print_categories")
    categories = page.categories
    for title in sorted(categories.keys()):
        logging.info("%s: %s" % (title, categories[title]))


class Wiki_Extract:

    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia('en')
        self.process = [preprocess_1, preprocess_2]

    def preprocess(self,target: str):
        return [target] + [f(target) for f in self.process if f(target) is not None]

    def get_property(self, target: str):
        targets = [target] + [f(target) for f in self.process if f(target) is not None]
        for target in targets:
            page = self.wiki.page(target.strip())
            if page.exists():
                return
                # result['attr'] = page.
        return result

    def get_summary(self, target: str):
        targets = [target] + [f(target) for f in self.process if f(target) is not None]
        for target in targets:
            page = self.wiki.page(target.strip())
            if page.exists() and page.summary is not None:
                return page.summary
        return ""

    def get_section(self, target: str):
        targets = [target] + [f(target) for f in self.process if f(target) is not None]
        for target in targets:
            page = self.wiki.page(target.strip())
            if page.exists() and page.sections is not None:
                return [s.title for s in page.sections]
        return None

    def get_categories(self, target: str):
        targets = [target] + [f(target) for f in self.process if f(target) is not None]
        for target in targets:
            page = self.wiki.page(target.strip())
            print(page)
            if page.exists() and page.categories is not None:
                title = [c.replace('Category:','') for c in page.categories.keys()]
                title = [c for c in title if preprocess_3(c) and target not in c]
                title.sort(key=len)
                if len(title) > 2: title = title[:2]
                return "__SEP__ ".join(title)
        return ""



if __name__ == "__main__":

    parser=argparse.ArgumentParser(description='Training')
    parser.add_argument('--target','-t',type=str)
    args=parser.parse_args()
    target = args.target

    wiki_extractor = Wiki_Extract()
    # print(wiki_extractor.get_section(target))
    # print(wiki_extractor.get_summary(target))
    cat = wiki_extractor.get_categories(target)
    cat.sort(key=len)
    print(cat)
