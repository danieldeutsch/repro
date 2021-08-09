import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

from repro.common import TemporaryDirectory, util
from repro.common.docker import make_volume_map, run_command
from repro.common.io import write_to_text_file
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.lin2004 import DEFAULT_IMAGE, MODEL_NAME
from repro.models.lin2004.commands import sentence_split

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-rouge")
class ROUGE(Model):
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        ngram_order: int = 4,
        porter_stemmer: bool = True,
        remove_stopwords: bool = False,
        sentence_split: bool = True,
        calculate_su4: bool = True,
    ):
        self.image = image
        self.ngram_order = ngram_order
        self.porter_stemmer = porter_stemmer
        self.remove_stopwords = remove_stopwords
        self.sentence_split = sentence_split
        self.calculate_su4 = calculate_su4

    def _maybe_sentence_split(self, texts: List[TextType]) -> List[List[str]]:
        if any(isinstance(text, str) for text in texts):
            if not all(isinstance(text, str) for text in texts):
                raise Exception(
                    f"Input texts are mixed types between strings and lists of strings. "
                    f"All must be of the same type"
                )
            return sentence_split(self.image, texts)
        else:
            return texts

    def _maybe_sentence_split_texts_list(
        self, texts_list: List[List[TextType]]
    ) -> List[List[List[str]]]:
        # Flatten the texts into a single list so we can call `_maybe_sentence_split`, then
        # rearrange the output to be parallel to `texts_list`
        flat_texts = []
        for texts in texts_list:
            flat_texts.extend(texts)

        split_texts = self._maybe_sentence_split(flat_texts)
        split_texts_list = []
        index = 0
        for texts in texts_list:
            split_texts_list.append([])
            for _ in texts:
                split_texts_list[-1].append(split_texts[index])
                index += 1
        return split_texts_list

    @staticmethod
    def _write_to_text_file(text: Union[str, List[str]], file_path: str) -> None:
        # We wrap the one text in a list, then write the whole list to a file. If
        # `text` is a `str`, then it will only write 1 line. If it's a `List[str]`,
        # it will be one sentence per line
        write_to_text_file([text], file_path, separator="\n")

    @staticmethod
    def _write_config_file(
        candidate_filenames_list: List[List[str]],
        reference_filenames_list: List[List[str]],
        file_path: str,
    ) -> None:
        with open(file_path, "w") as out:
            out.write(f'<ROUGE_EVAL version="1.0">\n')
            for i, (candidate_filenames, reference_filenames) in enumerate(
                zip(candidate_filenames_list, reference_filenames_list)
            ):
                out.write(f'<EVAL ID="{i + 1}">\n')
                out.write(f'<INPUT-FORMAT TYPE="SPL"></INPUT-FORMAT>\n')
                out.write(f"<PEER-ROOT>/</PEER-ROOT>\n")
                out.write(f"<MODEL-ROOT>/</MODEL-ROOT>\n")

                out.write(f"<PEERS>\n")
                for j, candidate_filename in enumerate(candidate_filenames):
                    out.write(f'<P ID="{j + 1}">{candidate_filename}</P>\n')
                out.write(f"</PEERS>\n")

                out.write(f"<MODELS>\n")
                for j, reference_filename in enumerate(reference_filenames):
                    symbol = chr(j + 65)
                    out.write(f'<M ID="{symbol}">{reference_filename}</M>\n')
                out.write(f"</MODELS>\n")

                out.write(f"</EVAL>\n")
            out.write(f"</ROUGE_EVAL>\n")

    @staticmethod
    def _parse_individual_line(
        columns: List[str],
    ) -> Tuple[int, int, float, float, float]:
        assert len(columns) == 7
        period = columns[3].index(".")
        group_index = int(columns[3][:period]) - 1
        candidate_index = int(columns[3][period + 1 :]) - 1
        recall = float(columns[4][2:]) * 100
        precision = float(columns[5][2:]) * 100
        f1 = float(columns[6][2:]) * 100
        return group_index, candidate_index, recall, precision, f1

    @staticmethod
    def _parse_stdout(stdout: str) -> Dict[int, Dict[int, Dict]]:
        lines = stdout.splitlines()
        # metrics_dict[group_index][candidate_index][metric] = value
        metrics_dict = defaultdict(lambda: defaultdict(dict))
        for line in lines:
            if line in [
                "---------------------------------------------",
                ".............................................",
            ]:
                continue
            columns = line.split()
            rouge_metric = columns[1].lower()
            if columns[2] == "Eval":
                (
                    group_index,
                    candidate_index,
                    recall,
                    precision,
                    f1,
                ) = ROUGE._parse_individual_line(columns)
                metrics_dict[group_index][candidate_index][rouge_metric] = {
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                }
        return metrics_dict

    def predict(
        self,
        candidate: TextType,
        references: List[TextType],
        **kwargs,
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "references": references}], **kwargs
        )[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[TextType, List[TextType]]]], **kwargs
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating ROUGE for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

        # ROUGE can be quite slow, so we deduplicate processing by grouping all of
        # the sources by identical targets. There could be duplicate references if you are scoring multiple
        # systems outputs at once. We then process the data grouped, and ungroup it at the end
        # so the output scores are parallel to the inputs.
        (
            grouped_candidates_list,
            grouped_references_list,
            group_mapping,
        ) = util.group_by_references(candidates, references_list)

        if self.sentence_split:
            grouped_candidates_list = self._maybe_sentence_split_texts_list(
                grouped_candidates_list
            )
            grouped_references_list = self._maybe_sentence_split_texts_list(
                grouped_references_list
            )

        # The ROUGE config file requires pointing to the candidate and reference filenames
        candidate_filenames_list = []
        reference_filenames_list = []

        with TemporaryDirectory() as temp:
            host_input_dir = f"{temp}/input"
            host_output_dir = f"{temp}/output"
            volume_map = make_volume_map(host_input_dir, host_output_dir)
            container_input_dir = volume_map[host_input_dir]

            # Serialize all of the texts to disk
            for i, (candidates, references) in enumerate(
                zip(grouped_candidates_list, grouped_references_list)
            ):
                candidate_filenames = []
                for j, candidate in enumerate(candidates):
                    candidate_filename = f"{i}/model.{j}.txt"
                    candidate_filenames.append(
                        f"{container_input_dir}/{candidate_filename}"
                    )
                    self._write_to_text_file(
                        candidate, f"{host_input_dir}/{candidate_filename}"
                    )
                candidate_filenames_list.append(candidate_filenames)

                reference_filenames = []
                for j, reference in enumerate(references):
                    symbol = chr(j + 65)
                    reference_filename = f"{i}/gold.{symbol}.txt"
                    reference_filenames.append(
                        f"{container_input_dir}/{reference_filename}"
                    )
                    self._write_to_text_file(
                        reference, f"{host_input_dir}/{reference_filename}"
                    )
                reference_filenames_list.append(reference_filenames)

            # Save the config file
            host_config_file = f"{host_input_dir}/config.xml"
            container_config_file = f"{container_input_dir}/config.xml"
            self._write_config_file(
                candidate_filenames_list, reference_filenames_list, host_config_file
            )

            # Run the command
            command = (
                f"perl ROUGE-1.5.5/ROUGE-1.5.5.pl"
                f"  -e ROUGE-1.5.5/data"
                f"  -n {self.ngram_order}"
                f"  -a"
                f"  -c 95"
                f"  -r 1000"
                f"  -p 0.5"
                f"  -t 0"
                f"  -d"
            )
            if self.porter_stemmer:
                command += " -m"
            if self.remove_stopwords:
                command += " -s"
            if self.calculate_su4:
                command += " -2 4 -u"
            command += f" {container_config_file}"

            stdout = run_command(
                self.image,
                command,
                volume_map=volume_map,
                network_disabled=True,
                stderr=False,
                silent=True,
            )

            metrics_dict = self._parse_stdout(stdout)
            micro_metrics = util.ungroup_values(metrics_dict, group_mapping)
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics
