import os


import os


def count_video(dir_path: str, recursive: bool = True) -> list[int]:
    """
    Возвращает список индексов [0, 1, 2, ..., N-1],
    где N — количество видеофайлов в папке.

    dir_path: путь к папке
    recursive: True — считать и в подпапках (os.walk), False — только в текущей (os.listdir)
    """
    video_exts = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".m4v")

    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Папка не найдена: {dir_path}")

    count = 0

    if recursive:
        for _, _, files in os.walk(dir_path):
            for name in files:
                if name.lower().endswith(video_exts):
                    count += 1
    else:
        for name in os.listdir(dir_path):
            full_path = os.path.join(dir_path, name)
            if os.path.isfile(full_path) and name.lower().endswith(video_exts):
                count += 1

    return list(range(count))


import os


def delete_video(
    keep_list: list[tuple[str, float]],
    remove_list: list[tuple[str, float]],
    folder_path: str,
    *,
    compare: str = "path",               # "path" или "path_time"
    dry_run: bool = False,               # True = ничего не удаляем (просто пропускаем os.remove)
    allow_outside_folder: bool = False,  # False = запрещаем удалять файлы вне folder_path
) -> None:
    """
    Принимает два массива кортежей (video_path, start_time) и путь к папке.

    Делает разницу: keep_list - remove_list, и удаляет из folder_path файлы,
    которые оказались в этой разнице.

    Ничего не возвращает.
    """

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Папка не найдена: {folder_path}")

    folder_abs = os.path.abspath(folder_path)

    # Нормализация путей для стабильного сравнения (учёт регистра/слешей и т.п.)
    def norm_path(p: str) -> str:
        return os.path.normcase(os.path.abspath(p))

    # Ключ сравнения для элемента (path, time)
    if compare == "path":
        def key(item: tuple[str, float]):
            return norm_path(item[0])
    elif compare == "path_time":
        def key(item: tuple[str, float]):
            return (norm_path(item[0]), round(float(item[1]), 1))
    else:
        raise ValueError('compare должен быть "path" или "path_time"')

    keep_keys = {key(x) for x in keep_list}
    remove_keys = {key(x) for x in remove_list}

    # Разница: есть в keep_list, но нет в remove_list
    diff_keys = keep_keys - remove_keys

    # Из diff_keys восстанавливаем список путей
    for dk in diff_keys:
        # при compare="path_time" dk будет tuple(path, time)
        p_norm = dk[0] if isinstance(dk, tuple) else dk
        p_abs = os.path.abspath(p_norm)

        # Защита: не удалять файлы вне folder_path
        if not allow_outside_folder:
            try:
                if os.path.commonpath([folder_abs, p_abs]) != folder_abs:
                    continue
            except ValueError:
                # например, разные диски на Windows
                continue

        # Удаляем только если это файл
        if os.path.isfile(p_abs):
            if not dry_run:
                os.remove(p_abs)


if __name__ == "__main__":
    print(count_video(r"C:\Users\user\Documents\Prodjeeeect\all_func\video_split"))
