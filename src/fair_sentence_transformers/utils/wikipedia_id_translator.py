import requests
from urllib.parse import urlparse, unquote
from typing import List, Dict, Optional, Union


class WikipediaCuridTranslator:
    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang

    def _extract_title_from_url(self, url: str) -> Optional[str]:
        try:
            path = urlparse(url).path
            if path.startswith("/wiki/"):
                return unquote(path[len("/wiki/") :])
        except Exception:
            pass
        return None

    def _batch_get_pageids_from_titles(
        self, titles: List[str], lang: str
    ) -> Dict[str, Optional[int]]:
        url = f"https://{lang}.wikipedia.org/w/api.php"
        result = {}
        for i in range(0, len(titles), 50):
            chunk = titles[i : i + 50]
            params = {"action": "query", "titles": "|".join(chunk), "format": "json"}
            resp = requests.get(url, params=params).json()
            pages = resp.get("query", {}).get("pages", {})
            for page in pages.values():
                title = page.get("title")
                pageid = page.get("pageid")
                if title:
                    result[title] = pageid
        return result

    def _batch_get_wikidata_qids(self, pageids: List[int]) -> Dict[int, Optional[str]]:
        url = f"https://{self.source_lang}.wikipedia.org/w/api.php"
        result = {}
        for i in range(0, len(pageids), 50):
            chunk = pageids[i : i + 50]
            params = {
                "action": "query",
                "prop": "pageprops",
                "pageids": "|".join(map(str, chunk)),
                "format": "json",
            }
            resp = requests.get(url, params=params).json()
            pages = resp.get("query", {}).get("pages", {})
            for pid in chunk:
                try:
                    result[pid] = pages[str(pid)]["pageprops"]["wikibase_item"]
                except KeyError:
                    result[pid] = None
        return result

    def _batch_get_target_titles(self, qids: List[str]) -> Dict[str, Optional[str]]:
        url = "https://www.wikidata.org/w/api.php"
        result = {}
        for i in range(0, len(qids), 50):
            chunk = qids[i : i + 50]
            params = {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "sitelinks",
                "format": "json",
            }
            resp = requests.get(url, params=params).json()
            entities = resp.get("entities", {})
            for qid in chunk:
                try:
                    result[qid] = entities[qid]["sitelinks"][f"{self.target_lang}wiki"][
                        "title"
                    ]
                except KeyError:
                    result[qid] = None
        return result

    def _batch_get_pageids_from_target_titles(
        self, titles: List[str]
    ) -> Dict[str, Optional[int]]:
        return self._batch_get_pageids_from_titles(titles, self.target_lang)

    def translate_id(
        self, src_id: int, return_urls: bool = False
    ) -> Optional[Union[int, str]]:
        result = self.translate_ids([src_id], return_urls=return_urls)
        return result.get(src_id)

    def translate_ids(
        self, src_ids: List[int], return_urls: bool = False
    ) -> Dict[int, Optional[Union[int, str]]]:
        pid_to_qid = self._batch_get_wikidata_qids(src_ids)
        qids = list({qid for qid in pid_to_qid.values() if qid})
        qid_to_title = self._batch_get_target_titles(qids)
        titles = list({title for title in qid_to_title.values() if title})
        title_to_pid = self._batch_get_pageids_from_target_titles(titles)

        result = {}
        for src_id in src_ids:
            qid = pid_to_qid.get(src_id)
            title = qid_to_title.get(qid) if qid else None
            if return_urls and title:
                result[src_id] = (
                    f"https://{self.target_lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                )
            else:
                result[src_id] = title_to_pid.get(title) if title else None
        return result

    def translate_url(
        self, url: str, return_urls: bool = False
    ) -> Optional[Union[int, str]]:
        return self.translate_urls([url], return_urls=return_urls).get(url)

    def translate_urls(
        self, urls: List[str], return_urls: bool = False
    ) -> Dict[str, Optional[Union[int, str]]]:
        # Step 0: Extract titles from URLs
        extracted_titles = [self._extract_title_from_url(url) for url in urls]
        title_to_url = {
            title: url for title, url in zip(extracted_titles, urls) if title
        }

        # Step 1: Get pageids from extracted titles
        title_to_srcid = self._batch_get_pageids_from_titles(
            [t for t in extracted_titles if t], self.source_lang
        )

        # Step 2: Translate pageids to target pageids or URLs
        src_ids = list(title_to_srcid.values())
        src_id_to_result = self.translate_ids(src_ids, return_urls=return_urls)

        # Step 3: Rebuild mapping from original URLs to results
        # We'll match via page ID instead of possibly changed titles
        pageid_to_url = {v: k for k, v in title_to_srcid.items()}
        result = {}
        for url in urls:
            orig_title = self._extract_title_from_url(url)
            src_id = title_to_srcid.get(
                orig_title.replace("_", " ")
            ) or title_to_srcid.get(orig_title)
            tgt_val = src_id_to_result.get(src_id) if src_id else None
            result[url] = tgt_val

        return result
